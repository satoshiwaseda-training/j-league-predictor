"""
scripts/label_improvement_test.py

2026/04/18-19の本命/最強層で外した3試合の根本原因分析に基づく
改修案A・Bの定量検証バックテスト。

検証ロジック:
  1. 2024-2025 (J1+J2) walk-forward 予測 → 統計モデル確率
  2. 各予測に対して classification (confidence_level, draw_alert) を計算
  3. 4種類の labeling 戦略を適用してラベル別正答率を比較
       baseline   : 現行 _get_strategy_label (high+draw_alert → 最強)
       改修B       : 最強に max_prob >= 0.65 の絶対下限を追加
       改修A      : confidence をシミュレートした Gemini大補正で downgrade
                    (本テストは履歴データに gemini delta がないため
                     簡易シミュレーションで近似)
       A+B 統合   : 両方を適用

Usage: python scripts/label_improvement_test.py [--division j1|j2|both]
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
    run_walk_forward,
)
from data_fetcher import get_multi_season_results


# ────────────────────────────────────────────────
# 1. classification (app.py._classify_prediction と同等)
# ────────────────────────────────────────────────
def classify_prediction(h_prob: float, d_prob: float, a_prob: float) -> dict:
    """probabilities (0-1 範囲) から confidence_level と draw_alert を返す"""
    h_pct = round(h_prob * 100)
    d_pct = round(d_prob * 100)
    a_pct = round(a_prob * 100)
    probs = sorted([h_pct, d_pct, a_pct], reverse=True)
    diff = probs[0] - probs[1]
    mx = probs[0]
    if diff >= 15:
        conf = "high"
    elif diff >= 5:
        conf = "medium"
    else:
        conf = "low"
    closeness = max(0.0, 1.0 - abs(h_prob - a_prob) * 2)
    draw_alert = (d_pct >= 25) and (closeness >= 0.5)
    return {
        "confidence": conf,
        "draw_alert": draw_alert,
        "max_prob": mx,
        "h_pct": h_pct, "d_pct": d_pct, "a_pct": a_pct,
        "diff": diff,
    }


# ────────────────────────────────────────────────
# 2. labeling 戦略
# ────────────────────────────────────────────────
def label_baseline(c: dict) -> str:
    """現行ロジック (app.py._get_strategy_label の主要分岐のみ)"""
    if c["confidence"] == "high":
        return "最強" if c["draw_alert"] else "本命"
    if c["confidence"] == "medium":
        return "波乱狙い" if c["draw_alert"] else "組み合わせ"
    return "スキップ" if c["draw_alert"] else "見送り"


def label_with_kaishu_b(c: dict, floor: float = 0.65) -> str:
    """改修B: 最強に max_prob >= floor*100 の絶対下限を追加"""
    if c["confidence"] == "high":
        if c["draw_alert"]:
            if c["max_prob"] >= floor * 100:
                return "最強"
            return "組み合わせ"  # 降格
        return "本命"
    if c["confidence"] == "medium":
        return "波乱狙い" if c["draw_alert"] else "組み合わせ"
    return "スキップ" if c["draw_alert"] else "見送り"


def label_with_kaishu_a(c: dict, h_form_w: int) -> str:
    """改修A 簡易版: ホーム直近5戦勝数 < 3 で high → medium に降格

    本来の改修A は Gemini pp_delta も条件に入れるが、履歴データには
    pp_delta が無いため、ここでは「確率の出どころに関わらず、ホームが
    弱フォームのとき高確信ラベルを下げる」という保守的な近似。
    """
    eff = dict(c)
    if eff["confidence"] == "high" and h_form_w < 3:
        eff["confidence"] = "medium"
    return label_baseline(eff)


def label_with_kaishu_ab(c: dict, h_form_w: int, floor: float = 0.65) -> str:
    """改修A+B 統合"""
    eff = dict(c)
    if eff["confidence"] == "high" and h_form_w < 3:
        eff["confidence"] = "medium"
    return label_with_kaishu_b(eff, floor=floor)


# ────────────────────────────────────────────────
# 3. 結果集計
# ────────────────────────────────────────────────
TIER_ORDER = ["最強", "本命", "波乱狙い", "組み合わせ", "スキップ", "見送り"]


def summarize(rows: list[dict], label_key: str) -> dict:
    by_label = defaultdict(lambda: {"n": 0, "hit": 0})
    for r in rows:
        lbl = r[label_key]
        by_label[lbl]["n"] += 1
        if r["pred_winner"] == r["actual"]:
            by_label[lbl]["hit"] += 1
    out = {}
    for lbl in TIER_ORDER:
        b = by_label.get(lbl, {"n": 0, "hit": 0})
        n = b["n"]
        out[lbl] = {
            "n": n,
            "hit": b["hit"],
            "acc": (b["hit"] / n) if n else None,
        }
    # 「賭ける」層 (最強+本命+波乱狙い+組み合わせ) の合算
    bet_n = sum(out[l]["n"] for l in ["最強", "本命", "波乱狙い", "組み合わせ"])
    bet_hit = sum(out[l]["hit"] for l in ["最強", "本命", "波乱狙い", "組み合わせ"])
    out["__BET_TOTAL__"] = {
        "n": bet_n,
        "hit": bet_hit,
        "acc": (bet_hit / bet_n) if bet_n else None,
    }
    # 高確信層 (最強+本命) の合算
    hi_n = sum(out[l]["n"] for l in ["最強", "本命"])
    hi_hit = sum(out[l]["hit"] for l in ["最強", "本命"])
    out["__HIGH_CONF__"] = {
        "n": hi_n,
        "hit": hi_hit,
        "acc": (hi_hit / hi_n) if hi_n else None,
    }
    return out


def print_summary(name: str, s: dict):
    print(f"\n  [{name}]")
    print(f"  {'label':<14} {'n':>5} {'hit':>5} {'acc':>8}")
    print(f"  {'-'*36}")
    for lbl in TIER_ORDER + ["__HIGH_CONF__", "__BET_TOTAL__"]:
        v = s[lbl]
        if v["n"] == 0 and lbl not in ("__HIGH_CONF__", "__BET_TOTAL__"):
            continue
        acc_s = f"{v['acc']*100:>6.1f}%" if v["acc"] is not None else "    --"
        print(f"  {lbl:<14} {v['n']:>5d} {v['hit']:>5d} {acc_s}")


# ────────────────────────────────────────────────
# 4. メインバックテスト
# ────────────────────────────────────────────────
def run_backtest(division: str, eval_season: int = 2025) -> list[dict]:
    print(f"\n>>> Loading {division.upper()} data ...")
    all_results = get_multi_season_results([2024, 2025], division)
    print(f"    {len(all_results)} matches loaded")

    print(f">>> Walk-forward eval={eval_season} (predictor=integrated)")
    res = run_walk_forward(
        all_results, eval_season=eval_season,
        predictor="integrated",
        weights=BASELINE_WEIGHTS, params=BASELINE_PARAMS,
    )
    preds = res["predictions"]
    print(f"    {len(preds)} predictions generated")

    # 予測ごとに classification と各ラベリング戦略を計算
    # h_form_w (ホーム直近5戦の勝数) を計算するため、results を再構築する
    from backtest_runner import rebuild_states
    rows = []
    for p in preds:
        idx = p["idx"]
        states = rebuild_states(all_results, idx)
        hs = states.get(p["home"])
        h_form = hs.get_form(5) if hs else []
        h_form_w = sum(1 for r in h_form if r == "W")

        c = classify_prediction(p["prob_home"], p["prob_draw"], p["prob_away"])
        rows.append({
            "date": p["date"],
            "section": p["section"],
            "home": p["home"], "away": p["away"],
            "actual": p["actual"], "pred_winner": p["predicted"],
            "prob_h": p["prob_home"], "prob_d": p["prob_draw"], "prob_a": p["prob_away"],
            "confidence": c["confidence"],
            "draw_alert": c["draw_alert"],
            "max_prob": c["max_prob"],
            "h_form_w5": h_form_w,
            "h_form": "".join(h_form),
            "label_baseline": label_baseline(c),
            "label_kaishu_b": label_with_kaishu_b(c),
            "label_kaishu_a": label_with_kaishu_a(c, h_form_w),
            "label_kaishu_ab": label_with_kaishu_ab(c, h_form_w),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--division", default="both", choices=["j1", "j2", "both"])
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--save", default="backtest_results/label_improvement.json")
    args = parser.parse_args()

    divisions = ["j1", "j2"] if args.division == "both" else [args.division]
    all_rows = []
    for div in divisions:
        rows = run_backtest(div, eval_season=args.season)
        for r in rows:
            r["division"] = div
        all_rows.extend(rows)

    print("\n" + "=" * 60)
    print(f"  RESULTS (eval={args.season}, n={len(all_rows)})")
    print("=" * 60)

    for strategy in ["label_baseline", "label_kaishu_b", "label_kaishu_a", "label_kaishu_ab"]:
        s = summarize(all_rows, strategy)
        print_summary(strategy, s)

    # 詳細保存
    out_path = Path(args.save)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    summary_obj = {
        strategy: summarize(all_rows, strategy)
        for strategy in ["label_baseline", "label_kaishu_b", "label_kaishu_a", "label_kaishu_ab"]
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "n": len(all_rows),
            "season": args.season,
            "divisions": divisions,
            "summary": summary_obj,
            "rows": all_rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved detail to {out_path}")

    # 最強層に絞った変化分析
    print("\n" + "=" * 60)
    print("  最強層の変化 (baseline → 改修B)")
    print("=" * 60)
    baseline_strong = [r for r in all_rows if r["label_baseline"] == "最強"]
    kaishu_strong = [r for r in all_rows if r["label_kaishu_b"] == "最強"]
    demoted = [r for r in baseline_strong if r["label_kaishu_b"] != "最強"]
    print(f"  baseline 最強: n={len(baseline_strong)}, hit={sum(1 for r in baseline_strong if r['pred_winner']==r['actual'])}")
    print(f"  改修B 最強:    n={len(kaishu_strong)}, hit={sum(1 for r in kaishu_strong if r['pred_winner']==r['actual'])}")
    print(f"  降格された試合: {len(demoted)} 件")
    print(f"    そのうち的中していた数: {sum(1 for r in demoted if r['pred_winner']==r['actual'])}")
    print(f"    そのうち外していた数:   {sum(1 for r in demoted if r['pred_winner']!=r['actual'])}")


if __name__ == "__main__":
    main()
