"""
scripts/upset_combo_analysis.py

「波乱狙い」ラベル試合に対して以下を比較:
  baseline: 第1推奨 (argmax) を単勝で買う
  combo_with_2nd: 第1推奨 + 第2推奨 (確率順、draw含むかは依存)
  combo_with_draw: 第1推奨 + draw の 2点組み合わせ (新提案)

2025 + 2026 J1+J2 walk-forward 予測 で集計。
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest_runner import BASELINE_WEIGHTS, BASELINE_PARAMS, run_walk_forward
from data_fetcher import get_multi_season_results
from label_improvement_test import classify_prediction


def collect_upset_layer(division: str, eval_season: int) -> list[dict]:
    """波乱狙いラベル (medium + draw_alert) の試合を抽出"""
    all_results = get_multi_season_results([2024, 2025], division)
    res = run_walk_forward(
        all_results, eval_season=eval_season,
        predictor="integrated",
        weights=BASELINE_WEIGHTS, params=BASELINE_PARAMS,
    )
    rows = []
    for p in res["predictions"]:
        c = classify_prediction(p["prob_home"], p["prob_draw"], p["prob_away"])
        if c["confidence"] != "medium" or not c["draw_alert"]:
            continue  # 波乱狙い以外は除外
        rows.append({
            "div": division, "date": p["date"],
            "home": p["home"], "away": p["away"],
            "actual": p["actual"], "pred_winner": p["predicted"],
            "h_pct": int(p["prob_home"]*100),
            "d_pct": int(p["prob_draw"]*100),
            "a_pct": int(p["prob_away"]*100),
        })
    return rows


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", default="2025")
    parser.add_argument("--divisions", default="j1,j2")
    args = parser.parse_args()
    seasons = [int(s) for s in args.seasons.split(",")]
    divs = args.divisions.split(",")

    rows = []
    for div in divs:
        for season in seasons:
            print(f"... loading {div} {season}", flush=True)
            rows.extend(collect_upset_layer(div, season))
    print(f"total upset rows: {len(rows)}", flush=True)

    # --- 戦略別評価 ---
    n = len(rows)
    if n == 0:
        print("データなし")
        return

    # baseline: 1st (argmax) のみ
    base_hit = sum(1 for r in rows if r["pred_winner"] == r["actual"])

    # combo_with_2nd: 1st + 2nd 確率の組合せ
    # 2nd は draw_alert 緩いと home/away の場合あり
    combo2_hit = 0
    # combo_with_draw: 1st + draw (1st が draw でなければ)
    combo_draw_hit = 0
    # 内訳カウント
    cases = {
        "actual=home": 0, "actual=draw": 0, "actual=away": 0,
        "1st=home": 0, "1st=draw": 0, "1st=away": 0,
    }
    for r in rows:
        cases[f"actual={r['actual']}"] += 1
        cases[f"1st={r['pred_winner']}"] += 1

        probs = sorted(
            [("home", r["h_pct"]), ("draw", r["d_pct"]), ("away", r["a_pct"])],
            key=lambda x: -x[1])
        first, second = probs[0][0], probs[1][0]

        # 1st + 2nd combo
        if r["actual"] in (first, second):
            combo2_hit += 1
        # 1st + draw combo
        if first == "draw":
            # 第1推奨が既にdraw なら 1st + 2nd と同じ
            if r["actual"] in (first, second):
                combo_draw_hit += 1
        else:
            if r["actual"] in (first, "draw"):
                combo_draw_hit += 1

    print(f"=== 波乱狙いラベル (medium + draw_alert) 検証: 2025+2026 J1+J2 ===")
    print(f"対象: n={n}")
    print()
    print(f"baseline (1st 単勝):                {base_hit:>3}/{n} = {base_hit/n*100:>5.1f}%")
    print(f"combo (1st + 2nd 確率順):            {combo2_hit:>3}/{n} = {combo2_hit/n*100:>5.1f}%  (draw 含むのは時々)")
    print(f"combo_draw (1st + draw 強制) [提案]:  {combo_draw_hit:>3}/{n} = {combo_draw_hit/n*100:>5.1f}%")
    print()
    print(f"提案効果 vs baseline: {(combo_draw_hit - base_hit)/n*100:+.1f}pp")
    print(f"提案効果 vs combo:    {(combo_draw_hit - combo2_hit)/n*100:+.1f}pp")
    print()
    print(f"内訳: {cases}")

    # ROI 簡易試算 (3-way 等オッズ仮定: home=2.5, draw=3.2, away=3.0)
    print()
    print("--- 単純 ROI 試算 (オッズ home=2.5/draw=3.2/away=3.0、各単位 0.5枚) ---")
    odds = {"home": 2.5, "draw": 3.2, "away": 3.0}
    # baseline: 1点 1枚で 1st を買う
    base_roi = 0
    for r in rows:
        if r["pred_winner"] == r["actual"]:
            base_roi += odds[r["actual"]] - 1
        else:
            base_roi -= 1
    # combo_draw: 1st 0.5枚 + draw 0.5枚
    combo_roi = 0
    for r in rows:
        probs = sorted([("home", r["h_pct"]), ("draw", r["d_pct"]), ("away", r["a_pct"])], key=lambda x: -x[1])
        first = probs[0][0]
        second_pick = "draw" if first != "draw" else probs[1][0]
        # 1st 0.5 + 2nd_pick 0.5
        bet = 1.0  # 合計1単位
        win = 0
        if r["actual"] == first:
            win += 0.5 * odds[first]
        if r["actual"] == second_pick:
            win += 0.5 * odds[second_pick]
        combo_roi += win - bet

    print(f"baseline  ROI : {base_roi/n*100:+5.1f}%/試合 (累計 {base_roi:+.1f})")
    print(f"combo_draw ROI: {combo_roi/n*100:+5.1f}%/試合 (累計 {combo_roi:+.1f})")
    print(f"差分: {(combo_roi - base_roi)/n*100:+.1f}pp/試合")

    # 保存
    out = Path("/sessions/magical-brave-heisenberg/mnt/j-league-predictor/backtest_results/upset_combo_analysis.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "n": n,
            "baseline_hit": base_hit,
            "combo2_hit": combo2_hit,
            "combo_draw_hit": combo_draw_hit,
            "baseline_acc": base_hit/n,
            "combo2_acc": combo2_hit/n,
            "combo_draw_acc": combo_draw_hit/n,
            "baseline_roi_per_game": base_roi/n,
            "combo_draw_roi_per_game": combo_roi/n,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
