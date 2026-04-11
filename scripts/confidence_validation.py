"""
scripts/confidence_validation.py - 確信度定義の有効性検証

旧ルール (max_prob) と新ルール (diff) を実データで比較:
- high/medium/low 別の正答率
- 単調性 (high > medium > low)
- draw_alert との掛け合わせ
- val=2025 と 2026 holdout

Usage:
    python scripts/confidence_validation.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import predict_logic
from data_fetcher import get_multi_season_results
from scripts.backtest_runner import rebuild_states, compute_ranks, build_elo
from scripts.predict_logic import (
    calculate_parameter_contributions, advantage_to_probs, compute_hybrid_v9,
)
from venues import get_venue_info


def classify_old(h: int, d: int, a: int, closeness: float) -> dict:
    """旧ルール: max_prob ベース (47/37)"""
    mx = max(h, d, a)
    if mx >= 47:
        conf = "high"
    elif mx >= 37:
        conf = "medium"
    else:
        conf = "low"
    draw_alert = d >= 25 and closeness >= 0.5
    return {"conf": conf, "draw_alert": draw_alert, "max": mx}


def classify_new(h: int, d: int, a: int, closeness: float) -> dict:
    """新ルール: diff (top1-top2) ベース (15/5)"""
    probs = sorted([h, d, a], reverse=True)
    top1, top2 = probs[0], probs[1]
    diff = top1 - top2
    if diff >= 15:
        conf = "high"
    elif diff >= 5:
        conf = "medium"
    else:
        conf = "low"
    draw_alert = d >= 25 and closeness >= 0.5
    return {"conf": conf, "draw_alert": draw_alert, "diff": diff}


def run_backtest(all_results: list, season: int) -> list[dict]:
    """hybrid_v9.1 で指定シーズンを逐次予測し、予測結果をリストで返す"""
    predictions = []
    for idx, match in enumerate(all_results):
        if match.get("season") != season:
            continue
        actual = match.get("winner")
        if not actual:
            continue
        home, away = match["home"], match["away"]

        states = rebuild_states(all_results, idx)
        hs = states.get(home)
        as_ = states.get(away)
        if not hs or not as_:
            continue
        ranks = compute_ranks(states)
        h_stats = hs.to_stats_dict(ranks.get(home, 99))
        a_stats = as_.to_stats_dict(ranks.get(away, 99))
        hf = hs.get_form(5)
        af = as_.get_form(5)
        hv = get_venue_info(home)
        av = get_venue_info(away)
        elo = build_elo(all_results, idx)
        eh, ea = elo.score_pair(home, away)

        c = calculate_parameter_contributions(
            home, away, h_stats, a_stats, hf, af,
            {}, {}, [], [], hv, av,
            elo_home_score=eh, elo_away_score=ea,
        )
        cl = c.get("closeness", 0.5)
        sh, sd, sa = advantage_to_probs(c["raw_home_advantage"], cl)
        raw_v7 = {"home_win_prob": sh, "draw_prob": sd, "away_win_prob": sa}
        hyb = compute_hybrid_v9(
            home, away, h_stats, a_stats, hf, af,
            v7_prediction=raw_v7, elo_home_score=eh, elo_away_score=ea,
        )
        h_p = hyb["home_win_prob"]
        d_p = hyb["draw_prob"]
        a_p = hyb["away_win_prob"]

        # 予測ラベル (argmax)
        if h_p >= a_p and h_p >= d_p:
            pred = "home"
        elif a_p > h_p and a_p >= d_p:
            pred = "away"
        else:
            pred = "draw"

        old = classify_old(h_p, d_p, a_p, cl)
        new = classify_new(h_p, d_p, a_p, cl)

        predictions.append({
            "idx": idx,
            "home": home, "away": away,
            "h": h_p, "d": d_p, "a": a_p,
            "closeness": cl,
            "actual": actual,
            "predicted": pred,
            "correct": pred == actual,
            "old_conf": old["conf"],
            "old_draw_alert": old["draw_alert"],
            "new_conf": new["conf"],
            "new_draw_alert": new["draw_alert"],
            "diff": new["diff"],
            "max_prob": old["max"],
        })
    return predictions


def bucket_accuracy(preds: list[dict], conf_key: str) -> dict:
    """確信度別正答率を集計"""
    from collections import defaultdict
    buckets = defaultdict(lambda: {"total": 0, "correct": 0})
    for p in preds:
        c = p[conf_key]
        buckets[c]["total"] += 1
        if p["correct"]:
            buckets[c]["correct"] += 1
    result = {}
    for k in ["high", "medium", "low"]:
        b = buckets.get(k, {"total": 0, "correct": 0})
        t = b["total"]
        cor = b["correct"]
        result[k] = {
            "total": t,
            "correct": cor,
            "accuracy": cor / t if t else 0.0,
        }
    return result


def cross_draw_alert(preds: list[dict], conf_key: str, draw_key: str) -> dict:
    """confidence x draw_alert クロス集計"""
    from collections import defaultdict
    grid = defaultdict(lambda: {"total": 0, "correct": 0})
    for p in preds:
        c = p[conf_key]
        da = "draw_alert" if p[draw_key] else "no_alert"
        key = f"{c}_{da}"
        grid[key]["total"] += 1
        if p["correct"]:
            grid[key]["correct"] += 1
    result = {}
    for c in ["high", "medium", "low"]:
        for da in ["draw_alert", "no_alert"]:
            k = f"{c}_{da}"
            b = grid.get(k, {"total": 0, "correct": 0})
            t = b["total"]
            result[k] = {
                "total": t,
                "correct": b["correct"],
                "accuracy": b["correct"] / t if t else 0.0,
            }
    return result


def monotonicity_score(bucket_acc: dict) -> dict:
    """単調性スコア: high > medium > low がどれだけ成立しているか"""
    h = bucket_acc["high"]["accuracy"]
    m = bucket_acc["medium"]["accuracy"]
    l = bucket_acc["low"]["accuracy"]
    return {
        "high_acc": round(h, 4),
        "med_acc": round(m, 4),
        "low_acc": round(l, 4),
        "h_minus_m": round(h - m, 4),
        "m_minus_l": round(m - l, 4),
        "h_minus_l": round(h - l, 4),
        "monotonic": h >= m >= l,  # 完全単調
        "monotonic_weak": (h >= m) or (m >= l),  # 部分的
    }


def print_section(title: str) -> None:
    print(f"\n{'='*78}")
    print(f"  {title}")
    print("=" * 78)


def print_bucket_table(old_bucket: dict, new_bucket: dict) -> None:
    print(f"{'Confidence':<10} {'旧 (n)':>8} {'旧 acc':>8} {'新 (n)':>8} {'新 acc':>8}")
    print("-" * 50)
    for k in ["high", "medium", "low"]:
        o = old_bucket[k]
        n = new_bucket[k]
        print(f"{k:<10} {o['total']:>8} {o['accuracy']:>8.4f} "
              f"{n['total']:>8} {n['accuracy']:>8.4f}")


def print_monotonicity(label: str, old_mono: dict, new_mono: dict) -> None:
    print(f"\n{label}")
    print(f"{'':16} {'旧 (max)':>12} {'新 (diff)':>12}")
    print("-" * 44)
    print(f"{'high acc':<16} {old_mono['high_acc']:>12.4f} {new_mono['high_acc']:>12.4f}")
    print(f"{'medium acc':<16} {old_mono['med_acc']:>12.4f} {new_mono['med_acc']:>12.4f}")
    print(f"{'low acc':<16} {old_mono['low_acc']:>12.4f} {new_mono['low_acc']:>12.4f}")
    print(f"{'h - m':<16} {old_mono['h_minus_m']:>+12.4f} {new_mono['h_minus_m']:>+12.4f}")
    print(f"{'m - l':<16} {old_mono['m_minus_l']:>+12.4f} {new_mono['m_minus_l']:>+12.4f}")
    print(f"{'h - l':<16} {old_mono['h_minus_l']:>+12.4f} {new_mono['h_minus_l']:>+12.4f}")
    print(f"{'単調性':<16} {str(old_mono['monotonic']):>12} {str(new_mono['monotonic']):>12}")


def main():
    print("Loading data...")
    all_results = get_multi_season_results([2024, 2025], "j1")
    print(f"Total: {len(all_results)}")

    final_report = {}

    for season in [2025, 2026]:
        print(f"\nRunning backtest on {season}...")
        preds = run_backtest(all_results, season)
        n = len(preds)
        correct = sum(1 for p in preds if p["correct"])
        overall = correct / n if n else 0
        print(f"  n={n}, overall acc={overall:.4f}")

        # 旧ルール
        old_bucket = bucket_accuracy(preds, "old_conf")
        old_mono = monotonicity_score(old_bucket)
        old_cross = cross_draw_alert(preds, "old_conf", "old_draw_alert")

        # 新ルール
        new_bucket = bucket_accuracy(preds, "new_conf")
        new_mono = monotonicity_score(new_bucket)
        new_cross = cross_draw_alert(preds, "new_conf", "new_draw_alert")

        print_section(f"Season {season} (n={n}, overall={overall:.4f})")

        print("\n--- 確信度別正答率 ---")
        print_bucket_table(old_bucket, new_bucket)

        print_monotonicity("単調性比較:", old_mono, new_mono)

        print("\n--- confidence x draw_alert クロス集計 ---")
        print(f"{'組合せ':<22} {'旧 (n)':>8} {'旧 acc':>8} {'新 (n)':>8} {'新 acc':>8}")
        print("-" * 62)
        for c in ["high", "medium", "low"]:
            for da in ["draw_alert", "no_alert"]:
                k = f"{c}_{da}"
                o = old_cross[k]
                nn = new_cross[k]
                label = f"{c:<8} {da}"
                print(f"{label:<22} {o['total']:>8} {o['accuracy']:>8.4f} "
                      f"{nn['total']:>8} {nn['accuracy']:>8.4f}")

        final_report[season] = {
            "n": n,
            "overall_accuracy": round(overall, 4),
            "old": {
                "bucket": old_bucket,
                "monotonicity": old_mono,
                "cross_draw": old_cross,
            },
            "new": {
                "bucket": new_bucket,
                "monotonicity": new_mono,
                "cross_draw": new_cross,
            },
        }

    # 保存
    out_dir = Path(__file__).parent.parent / "backtest_results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "confidence_validation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
