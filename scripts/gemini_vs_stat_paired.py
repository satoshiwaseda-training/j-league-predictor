"""
scripts/gemini_vs_stat_paired.py

Gemini オン / オフ の同一試合での head-to-head 比較。

手順:
  1. data/predictions.json から Gemini 予測 (model=='gemini-*') を抽出
  2. data_fetcher から実結果を取得して照合
  3. 同じ試合に対して walk-forward 統計モデル (integrated, Gemini 非使用)
     で「その試合時点までの情報のみで」予測を再生成
  4. Gemini と統計の argmax / Brier / logloss / 各カテゴリ的中を比較

ライブ予測のGemini 呼び出しを再現することはできないが (LLM非決定的で
当時と同じ応答を再現できない)、保存された Gemini 予測は実際にライブで
生成されたものなので、これを固定として統計モデルと比べれば
「Gemini を入れたことで良くなったか悪くなったか」を直接計測できる。
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest_runner import (
    BASELINE_WEIGHTS, BASELINE_PARAMS,
    predict_integrated, build_elo, rebuild_states, compute_ranks,
)
from data_fetcher import get_past_results, get_historical_results


def load_gemini_preds() -> list[dict]:
    d = json.load(open("data/predictions.json"))
    return [p for p in d["predictions"]
            if p.get("prediction", {}).get("model", "").startswith("gemini")]


def load_all_results(divisions: list[str]) -> list[dict]:
    """2024+2025+2026 (現シーズン) の試合結果を統合"""
    all_r = []
    for div in divisions:
        for y in [2024, 2025]:
            all_r.extend(get_historical_results(y, div))
        all_r.extend(get_past_results(div))
    for r in all_r:
        if "season" not in r:
            # current season entries may miss season field
            r["season"] = int(r["date"][:4])
    # dedupe by (date, home, away, division)
    seen = set()
    uniq = []
    for r in all_r:
        k = (r.get("date"), r.get("home"), r.get("away"), r.get("division"))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    return sorted(uniq, key=lambda r: r["date"])


def brier_from(p: dict, actual: str) -> float:
    y = {c: (1.0 if c == actual else 0.0) for c in ["home", "draw", "away"]}
    return sum((p[c] - y[c])**2 for c in p)


def logloss_from(p: dict, actual: str) -> float:
    return -math.log(max(p[actual], 0.01))


def run_stat_prediction(idx: int, all_results: list[dict], match: dict) -> dict:
    """walk-forward: その試合時点までの情報のみで統計モデル予測"""
    from venues import get_venue_info

    home, away = match["home"], match["away"]
    states = rebuild_states(all_results, idx)
    ranks = compute_ranks(states)
    hs = states.get(home)
    as_ = states.get(away)
    if not hs or not as_ or hs.games == 0 or as_.games == 0:
        return None
    h_stats = hs.to_stats_dict(ranks.get(home, 99))
    a_stats = as_.to_stats_dict(ranks.get(away, 99))
    h_form = hs.get_form(5)
    a_form = as_.get_form(5)
    elo = build_elo(all_results, idx)
    probs = predict_integrated(
        home, away, h_stats, a_stats, h_form, a_form,
        BASELINE_WEIGHTS, BASELINE_PARAMS, elo,
    )
    return probs


def main():
    print(">>> Loading all results (J1+J2, 2024-2026)")
    all_results = load_all_results(["j1", "j2"])
    print(f"    {len(all_results)} total matches")

    # (date, home, away) -> (idx, winner)
    by_match = {}
    for i, r in enumerate(all_results):
        k = (r["date"], r["home"], r["away"])
        by_match[k] = (i, r.get("winner"), r.get("home_score"), r.get("away_score"))

    print(">>> Loading Gemini predictions from predictions.json")
    gemini_preds = load_gemini_preds()
    print(f"    {len(gemini_preds)} Gemini predictions")

    # 照合できるものだけ残す
    paired_rows = []
    for gp in gemini_preds:
        m = gp["match"]
        k = (m["date"], m["home"], m["away"])
        if k not in by_match:
            continue
        idx, winner, hs, as_ = by_match[k]
        if not winner:
            continue
        stat_probs = run_stat_prediction(idx, all_results, m)
        if stat_probs is None:
            continue
        gp_pred = gp["prediction"]
        gem_probs = {
            "home": gp_pred["home_win_prob"] / 100,
            "draw": gp_pred["draw_prob"] / 100,
            "away": gp_pred["away_win_prob"] / 100,
        }
        # normalize
        s = sum(gem_probs.values())
        gem_probs = {k: v/s for k, v in gem_probs.items()}
        paired_rows.append({
            "date": m["date"], "home": m["home"], "away": m["away"],
            "actual": winner, "score": f"{hs}-{as_}",
            "gem": gem_probs,
            "stat": stat_probs,
            "gem_argmax": max(gem_probs, key=gem_probs.get),
            "stat_argmax": max(stat_probs, key=stat_probs.get),
        })
    print(f"    paired: {len(paired_rows)}")

    if not paired_rows:
        print("No paired data")
        return

    # ─── 集計 ───
    print("\n" + "=" * 80)
    print("  Gemini ON vs OFF (Stat-only) 同一試合 head-to-head")
    print("=" * 80)

    def metrics(rows, probs_key, argmax_key):
        n = len(rows)
        hit = sum(1 for r in rows if r[argmax_key] == r["actual"])
        brier = sum(brier_from(r[probs_key], r["actual"]) for r in rows) / n
        ll = sum(logloss_from(r[probs_key], r["actual"]) for r in rows) / n
        return {"n": n, "hit": hit, "acc": hit/n, "brier": brier, "logloss": ll}

    gem_met = metrics(paired_rows, "gem", "gem_argmax")
    stat_met = metrics(paired_rows, "stat", "stat_argmax")

    print(f"\n  [Gemini ON]   n={gem_met['n']:>3}  acc={gem_met['acc']*100:>5.1f}% ({gem_met['hit']}/{gem_met['n']})  Brier={gem_met['brier']:.4f}  logloss={gem_met['logloss']:.4f}")
    print(f"  [Gemini OFF]  n={stat_met['n']:>3}  acc={stat_met['acc']*100:>5.1f}% ({stat_met['hit']}/{stat_met['n']})  Brier={stat_met['brier']:.4f}  logloss={stat_met['logloss']:.4f}")
    print(f"  [Δ (ON-OFF)]  acc={((gem_met['acc']-stat_met['acc'])*100):+.1f}pp  Brier={(gem_met['brier']-stat_met['brier']):+.4f}  logloss={(gem_met['logloss']-stat_met['logloss']):+.4f}")

    # ─── 試合別の変化 ───
    print("\n" + "=" * 80)
    print("  試合別: Gemini と Stat の argmax 比較")
    print("=" * 80)
    print(f"  {'date':<11} {'home':<18} {'vs':<18} {'actual':<5} "
          f"{'Gem':<14} {'Stat':<14} {'GemHit':<6} {'StatHit':<7} {'Δ':<6}")
    counts = {"gem_only_hit": 0, "stat_only_hit": 0, "both": 0, "neither": 0}
    for r in paired_rows:
        g_txt = f"{int(r['gem']['home']*100)}/{int(r['gem']['draw']*100)}/{int(r['gem']['away']*100)} → {r['gem_argmax'][0]}"
        s_txt = f"{int(r['stat']['home']*100)}/{int(r['stat']['draw']*100)}/{int(r['stat']['away']*100)} → {r['stat_argmax'][0]}"
        gh = r['gem_argmax'] == r['actual']
        sh = r['stat_argmax'] == r['actual']
        delta = "Gem+" if gh and not sh else "Stat+" if sh and not gh else "=" if gh == sh else ""
        if gh and not sh: counts["gem_only_hit"] += 1
        if sh and not gh: counts["stat_only_hit"] += 1
        if gh and sh: counts["both"] += 1
        if not gh and not sh: counts["neither"] += 1
        print(f"  {r['date']:<11} {r['home'][:16]:<18} {r['away'][:16]:<18} "
              f"{r['actual']:<5} {g_txt:<14} {s_txt:<14} "
              f"{'○' if gh else '✗':<6} {'○' if sh else '✗':<7} {delta:<6}")

    print("\n  変化内訳:")
    print(f"    両方的中:         {counts['both']}")
    print(f"    Gemini だけ的中:   {counts['gem_only_hit']}")
    print(f"    Stat だけ的中:     {counts['stat_only_hit']}")
    print(f"    両方外し:         {counts['neither']}")

    # ─── 高確信層に限定した比較 ───
    # Gemini の max_prob >= 50 の試合で Stat がどうだったか
    print("\n" + "=" * 80)
    print("  Gemini が高確信 (max_prob >= 50) な試合だけ抽出")
    print("=" * 80)
    hi = [r for r in paired_rows if max(r['gem'].values()) >= 0.50]
    print(f"  該当試合: {len(hi)}")
    if hi:
        gm_hi = metrics(hi, "gem", "gem_argmax")
        st_hi = metrics(hi, "stat", "stat_argmax")
        print(f"    Gemini acc={gm_hi['acc']*100:.1f}%  Brier={gm_hi['brier']:.4f}  logloss={gm_hi['logloss']:.4f}")
        print(f"    Stat   acc={st_hi['acc']*100:.1f}%  Brier={st_hi['brier']:.4f}  logloss={st_hi['logloss']:.4f}")

    # 保存
    out = Path("backtest_results/gemini_vs_stat_paired.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "n": len(paired_rows),
            "gemini_on": gem_met,
            "gemini_off": stat_met,
            "delta": {
                "acc": gem_met["acc"] - stat_met["acc"],
                "brier": gem_met["brier"] - stat_met["brier"],
                "logloss": gem_met["logloss"] - stat_met["logloss"],
            },
            "counts": counts,
            "rows": paired_rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved detail to {out}")


if __name__ == "__main__":
    main()
