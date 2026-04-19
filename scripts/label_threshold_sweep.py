"""
scripts/label_threshold_sweep.py

label_improvement_test を拡張: 改修A の form 閾値、改修B の絶対確率
下限を複数試して、最も成績の上がる組み合わせを探す。

評価対象:
  J1 2025 + J2 2025 + (利用可能なら) 2026 walk-forward

採用基準:
  - 高確信層 (最強+本命) の正答率 が baseline を超えること
  - かつ高確信層の n が極端に小さくならないこと (n>=20 を目安)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest_runner import (
    BASELINE_WEIGHTS, BASELINE_PARAMS,
    run_walk_forward, rebuild_states,
)
from data_fetcher import get_multi_season_results
from label_improvement_test import (
    classify_prediction, label_baseline,
    label_with_kaishu_b, label_with_kaishu_a, label_with_kaishu_ab,
)


def label_kb_floor(c: dict, floor: float) -> str:
    """改修B with arbitrary floor"""
    if c["confidence"] == "high":
        if c["draw_alert"]:
            if c["max_prob"] >= floor * 100:
                return "最強"
            return "組み合わせ"
        return "本命"
    if c["confidence"] == "medium":
        return "波乱狙い" if c["draw_alert"] else "組み合わせ"
    return "スキップ" if c["draw_alert"] else "見送り"


def label_ka_w(c: dict, h_form_w: int, w_min: int) -> str:
    """改修A with arbitrary w_min"""
    eff = dict(c)
    if eff["confidence"] == "high" and h_form_w < w_min:
        eff["confidence"] = "medium"
    return label_baseline(eff)


def label_ka_w_kb(c: dict, h_form_w: int, w_min: int, floor: float) -> str:
    """改修A+B 統合"""
    eff = dict(c)
    if eff["confidence"] == "high" and h_form_w < w_min:
        eff["confidence"] = "medium"
    return label_kb_floor(eff, floor=floor)


def collect_rows(division: str, eval_seasons: list[int]) -> list[dict]:
    """walk-forward 予測 → 行データに classification と form を付加"""
    train_seasons = [s - 1 for s in eval_seasons if s - 1 >= 2024]
    all_seasons = sorted(set([2024] + eval_seasons))
    print(f">>> {division.upper()}: loading seasons {all_seasons}")
    all_results = get_multi_season_results(all_seasons, division)
    print(f"    {len(all_results)} matches loaded")

    rows = []
    for eval_season in eval_seasons:
        print(f"    eval season={eval_season} (predictor=integrated)")
        res = run_walk_forward(
            all_results, eval_season=eval_season,
            predictor="integrated",
            weights=BASELINE_WEIGHTS, params=BASELINE_PARAMS,
        )
        preds = res["predictions"]
        print(f"      → {len(preds)} predictions")
        for p in preds:
            idx = p["idx"]
            states = rebuild_states(all_results, idx)
            hs = states.get(p["home"])
            h_form = hs.get_form(5) if hs else []
            h_form_w = sum(1 for r in h_form if r == "W")
            c = classify_prediction(p["prob_home"], p["prob_draw"], p["prob_away"])
            rows.append({
                "season": eval_season,
                "division": division,
                "date": p["date"],
                "home": p["home"], "away": p["away"],
                "actual": p["actual"], "pred_winner": p["predicted"],
                "prob_h": p["prob_home"], "prob_d": p["prob_draw"], "prob_a": p["prob_away"],
                "confidence": c["confidence"],
                "draw_alert": c["draw_alert"],
                "max_prob": c["max_prob"],
                "h_form_w5": h_form_w,
            })
    return rows


def evaluate(rows: list[dict], label_fn) -> dict:
    """ラベル関数を適用して、各層の正答率と n を返す"""
    by_label = defaultdict(lambda: {"n": 0, "hit": 0})
    for r in rows:
        c = {
            "confidence": r["confidence"],
            "draw_alert": r["draw_alert"],
            "max_prob": r["max_prob"],
        }
        lbl = label_fn(c, r["h_form_w5"])
        by_label[lbl]["n"] += 1
        if r["pred_winner"] == r["actual"]:
            by_label[lbl]["hit"] += 1
    out = {l: dict(v, acc=(v["hit"]/v["n"]) if v["n"] else None)
           for l, v in by_label.items()}
    # 高確信層集計
    hi_n = sum(by_label[l]["n"] for l in ["最強", "本命"])
    hi_h = sum(by_label[l]["hit"] for l in ["最強", "本命"])
    bet_n = sum(by_label[l]["n"] for l in ["最強", "本命", "波乱狙い", "組み合わせ"])
    bet_h = sum(by_label[l]["hit"] for l in ["最強", "本命", "波乱狙い", "組み合わせ"])
    out["__HIGH__"] = {"n": hi_n, "hit": hi_h, "acc": (hi_h/hi_n) if hi_n else None}
    out["__BET__"]  = {"n": bet_n, "hit": bet_h, "acc": (bet_h/bet_n) if bet_n else None}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", default="2025", help="comma-sep seasons (e.g., '2025,2026')")
    parser.add_argument("--save", default="backtest_results/threshold_sweep.json")
    args = parser.parse_args()

    eval_seasons = [int(s) for s in args.seasons.split(",")]
    rows = []
    for div in ["j1", "j2"]:
        rows.extend(collect_rows(div, eval_seasons))
    print(f"\nTotal rows for evaluation: {len(rows)}")

    # ─── ベースライン ───
    print("\n" + "=" * 70)
    print("  BASELINE (現行ロジック)")
    print("=" * 70)
    base = evaluate(rows, lambda c, fw: label_baseline(c))
    print(f"  最強     : n={base.get('最強',{}).get('n',0):>3}  hit={base.get('最強',{}).get('hit',0):>3}  acc={(base.get('最強',{}).get('acc') or 0)*100:>5.1f}%")
    print(f"  本命     : n={base.get('本命',{}).get('n',0):>3}  hit={base.get('本命',{}).get('hit',0):>3}  acc={(base.get('本命',{}).get('acc') or 0)*100:>5.1f}%")
    print(f"  HIGH合算 : n={base['__HIGH__']['n']:>3}  hit={base['__HIGH__']['hit']:>3}  acc={base['__HIGH__']['acc']*100:>5.1f}%")
    print(f"  BET合算  : n={base['__BET__']['n']:>3}  hit={base['__BET__']['hit']:>3}  acc={base['__BET__']['acc']*100:>5.1f}%")

    base_hi_acc = base['__HIGH__']['acc']
    base_hi_n = base['__HIGH__']['n']

    # ─── 改修A 単体スイープ ───
    print("\n" + "=" * 70)
    print("  改修A 単体スイープ (form W数下限を変える)")
    print("=" * 70)
    print(f"  {'w_min':<7} {'最強n':>5} {'最強acc':>8} {'本命n':>5} {'本命acc':>8} {'HIGHn':>6} {'HIGHacc':>8} {'Δacc':>7}")
    sweep_a = {}
    for w_min in [1, 2, 3, 4]:
        r = evaluate(rows, lambda c, fw, w=w_min: label_ka_w(c, fw, w))
        sweep_a[w_min] = r
        ms = r.get('最強', {"n":0,"hit":0,"acc":None})
        bs = r.get('本命', {"n":0,"hit":0,"acc":None})
        hi = r['__HIGH__']
        d = (hi['acc'] - base_hi_acc)*100 if (hi['acc'] is not None and base_hi_acc is not None) else 0
        print(f"  w<{w_min:<5} {ms['n']:>5} {(ms['acc'] or 0)*100:>7.1f}% {bs['n']:>5} {(bs['acc'] or 0)*100:>7.1f}% "
              f"{hi['n']:>6} {(hi['acc'] or 0)*100:>7.1f}% {d:>+6.1f}pp")

    # ─── 改修B 単体スイープ (整合性確認のみ; 統計モデルでは発火せず) ───
    print("\n" + "=" * 70)
    print("  改修B 単体スイープ (最強の絶対確率下限)")
    print("=" * 70)
    print(f"  {'floor':<7} {'最強n':>5} {'最強acc':>8} {'HIGHn':>6} {'HIGHacc':>8} {'Δacc':>7}")
    for floor in [0.50, 0.55, 0.60, 0.65]:
        r = evaluate(rows, lambda c, fw, fl=floor: label_kb_floor(c, fl))
        ms = r.get('最強', {"n":0,"hit":0,"acc":None})
        hi = r['__HIGH__']
        d = (hi['acc'] - base_hi_acc)*100 if (hi['acc'] is not None and base_hi_acc is not None) else 0
        print(f"  {floor:<7.2f} {ms['n']:>5} {(ms['acc'] or 0)*100:>7.1f}% {hi['n']:>6} {(hi['acc'] or 0)*100:>7.1f}% {d:>+6.1f}pp")

    # ─── 改修A+B 統合スイープ ───
    print("\n" + "=" * 70)
    print("  改修A+B 統合スイープ")
    print("=" * 70)
    print(f"  {'w_min':<5} {'floor':<6} {'最強n':>5} {'最強acc':>8} {'本命n':>5} {'本命acc':>8} {'HIGHn':>6} {'HIGHacc':>8} {'Δacc':>7}")
    sweep_ab = []
    for w_min in [1, 2, 3]:
        for floor in [0.50, 0.55, 0.60, 0.65]:
            r = evaluate(rows, lambda c, fw, w=w_min, fl=floor: label_ka_w_kb(c, fw, w, fl))
            ms = r.get('最強', {"n":0,"hit":0,"acc":None})
            bs = r.get('本命', {"n":0,"hit":0,"acc":None})
            hi = r['__HIGH__']
            d = (hi['acc'] - base_hi_acc)*100 if (hi['acc'] is not None and base_hi_acc is not None) else 0
            sweep_ab.append({
                "w_min": w_min, "floor": floor,
                "saigo_n": ms['n'], "saigo_acc": ms['acc'],
                "honmei_n": bs['n'], "honmei_acc": bs['acc'],
                "high_n": hi['n'], "high_acc": hi['acc'],
                "delta": d,
            })
            print(f"  w<{w_min:<3} {floor:<6.2f} {ms['n']:>5} {(ms['acc'] or 0)*100:>7.1f}% "
                  f"{bs['n']:>5} {(bs['acc'] or 0)*100:>7.1f}% "
                  f"{hi['n']:>6} {(hi['acc'] or 0)*100:>7.1f}% {d:>+6.1f}pp")

    # ─── 推奨モデル選定 ───
    print("\n" + "=" * 70)
    print("  推奨モデル選定 (HIGHacc 最大化、ただし HIGHn>=20)")
    print("=" * 70)
    candidates = [s for s in sweep_ab if s["high_n"] >= 20 and s["high_acc"] is not None]
    candidates.sort(key=lambda s: -s["high_acc"])
    for i, s in enumerate(candidates[:5]):
        print(f"  #{i+1}: w<{s['w_min']} floor={s['floor']:.2f}  "
              f"HIGH n={s['high_n']} acc={s['high_acc']*100:.1f}% (Δ{s['delta']:+.1f}pp)")

    # 単体ベスト改修Aも候補に
    sweep_a_clean = []
    for w_min, r in sweep_a.items():
        hi = r['__HIGH__']
        if hi['n'] >= 20 and hi['acc'] is not None:
            sweep_a_clean.append({
                "name": f"改修A単体 w<{w_min}",
                "high_n": hi['n'], "high_acc": hi['acc'],
                "delta": (hi['acc'] - base_hi_acc) * 100,
            })
    sweep_a_clean.sort(key=lambda s: -s["high_acc"])

    print("\n  改修A 単体での候補:")
    for s in sweep_a_clean:
        print(f"    {s['name']}  HIGH n={s['high_n']} acc={s['high_acc']*100:.1f}% (Δ{s['delta']:+.1f}pp)")

    out_path = Path(args.save)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_total": len(rows),
            "seasons": eval_seasons,
            "baseline": {"high_n": base_hi_n, "high_acc": base_hi_acc},
            "sweep_AB": sweep_ab,
            "sweep_A_alone": sweep_a_clean,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved sweep to {out_path}")


if __name__ == "__main__":
    main()
