"""
scripts/gemini_contribution_test.py

Gemini補正が実際に精度を改善しているかを検証。

方法:
  - predictions.json から Gemini 予測 と statistical-only 予測 を抽出
  - 各予測について、data_fetcher で同じ試合の実結果を照合
  - accuracy / Brier / log_loss を比較

このテストは 2026年 3月のデータが対象。
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_fetcher import get_multi_season_results


def load_preds():
    d = json.load(open("data/predictions.json"))
    return d["predictions"]


def build_result_index():
    """(date, home, away) -> winner のインデックスを作る"""
    from data_fetcher import get_past_results, get_historical_results
    idx = {}
    for div in ["j1", "j2"]:
        # 現シーズン (2026) と過去 (2025, 2024) を全部マージ
        results = list(get_past_results(div))
        for y in [2025, 2024]:
            results.extend(get_historical_results(y, div))
        for r in results:
            if not r.get("home") or not r.get("away"):
                continue
            key = (r["date"], r["home"], r["away"])
            idx[key] = r.get("winner")
    return idx


def compute_metrics(rows: list[dict]) -> dict:
    """各行は {h, d, a, actual, pred_winner}"""
    n = len(rows)
    if n == 0:
        return {"n": 0}
    hit = sum(1 for r in rows if r["pred_winner"] == r["actual"])
    # Brier
    brier_sum = 0.0
    logloss_sum = 0.0
    for r in rows:
        p = {"home": r["h"]/100, "draw": r["d"]/100, "away": r["a"]/100}
        # normalize
        t = sum(p.values())
        p = {k: v/t for k, v in p.items()}
        # Brier: sum((p - y)^2)
        y = {k: (1.0 if k == r["actual"] else 0.0) for k in ["home", "draw", "away"]}
        brier_sum += sum((p[k] - y[k])**2 for k in p)
        # logloss
        p_true = max(p[r["actual"]], 0.01)
        logloss_sum += -math.log(p_true)
    return {
        "n": n,
        "acc": hit / n,
        "hit": hit,
        "brier": brier_sum / n,
        "logloss": logloss_sum / n,
    }


def main():
    preds = load_preds()
    result_idx = build_result_index()
    print(f"loaded {len(preds)} predictions, {len(result_idx)} historical results indexed")

    # model 別に予測を分類
    by_model = defaultdict(list)
    for p in preds:
        model = p.get("prediction", {}).get("model", "?")
        date = p["match"]["date"]
        home = p["match"]["home"]
        away = p["match"]["away"]
        actual = result_idx.get((date, home, away))
        if actual is None:
            # 実結果が見つからない→除外
            continue
        pred = p["prediction"]
        by_model[model].append({
            "date": date, "home": home, "away": away,
            "h": pred["home_win_prob"],
            "d": pred["draw_prob"],
            "a": pred["away_win_prob"],
            "actual": actual,
            "pred_winner": pred["pred_winner"],
            "max_prob": max(pred["home_win_prob"], pred["draw_prob"], pred["away_win_prob"]),
        })

    print("\n" + "=" * 70)
    print("  モデル別の実結果との照合結果")
    print("=" * 70)
    for model, rows in sorted(by_model.items()):
        m = compute_metrics(rows)
        print(f"\n  [{model}] n={m['n']}")
        if m["n"] == 0:
            continue
        print(f"    accuracy:  {m['acc']*100:.1f}% ({m['hit']}/{m['n']})")
        print(f"    Brier:     {m['brier']:.4f}")
        print(f"    logloss:   {m['logloss']:.4f}")

    # 同一試合で Gemini と stat 両方あるペアを比較
    print("\n" + "=" * 70)
    print("  同一試合での Gemini vs Stat 直接比較 (ペア分析)")
    print("=" * 70)
    by_match = defaultdict(dict)
    for model, rows in by_model.items():
        for r in rows:
            key = (r["date"], r["home"], r["away"])
            by_match[key][model] = r
    paired = {k: v for k, v in by_match.items() if len(v) >= 2}
    print(f"  両方の model で予測がある試合: {len(paired)} 件")

    if paired:
        # Gemini と Stat-only のペアがある試合だけ抽出
        gemini_paired = []
        stat_paired = []
        for k, v in paired.items():
            gm = v.get("gemini-2.0-flash")
            so = v.get("statistical-only") or v.get("statistical-backtest")
            if gm and so:
                gemini_paired.append(gm)
                stat_paired.append(so)
        if gemini_paired:
            print(f"\n  Gemini + Stat のペア: {len(gemini_paired)} 件")
            gm_met = compute_metrics(gemini_paired)
            st_met = compute_metrics(stat_paired)
            print(f"    Gemini        acc={gm_met['acc']*100:.1f}% Brier={gm_met['brier']:.4f} logloss={gm_met['logloss']:.4f}")
            print(f"    Stat (same n) acc={st_met['acc']*100:.1f}% Brier={st_met['brier']:.4f} logloss={st_met['logloss']:.4f}")

            # どの試合で Gemini が改善/悪化したか
            print("\n  試合別の比較:")
            print(f"    {'date':<12} {'home':<20} {'vs':<20} {'actual':<5} {'Gem prob→pred':<20} {'Stat prob→pred':<20} {'Δcorrect':<8}")
            gem_hits = 0
            stat_hits = 0
            for gm, st in zip(gemini_paired, stat_paired):
                gh = f"H{gm['h']}D{gm['d']}A{gm['a']}→{gm['pred_winner'][0]}"
                sh = f"H{st['h']}D{st['d']}A{st['a']}→{st['pred_winner'][0]}"
                gm_c = gm["pred_winner"] == gm["actual"]
                st_c = st["pred_winner"] == st["actual"]
                if gm_c: gem_hits += 1
                if st_c: stat_hits += 1
                delta = ("+" if gm_c and not st_c else "-" if st_c and not gm_c else "=")
                print(f"    {gm['date']:<12} {gm['home'][:18]:<20} {gm['away'][:18]:<20} "
                      f"{gm['actual'][:4]:<5} {gh:<20} {sh:<20} {delta}")
            print(f"\n  ペア正答: Gemini {gem_hits}/{len(gemini_paired)} vs Stat {stat_hits}/{len(stat_paired)}")


if __name__ == "__main__":
    main()
