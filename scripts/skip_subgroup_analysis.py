"""
scripts/skip_subgroup_analysis.py

スキップ層 (confidence=low, draw_alert=True) の試合を特徴量別に分解して、
ランダム (~33%) を有意に超える正答率を持つサブグループを探す。

見つかれば「スキップの中でも勝ちやすい試合」として別ラベル昇格を検討できる。

Usage:
  python scripts/skip_subgroup_analysis.py --seasons 2025
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
    run_walk_forward, rebuild_states, build_elo,
)
from data_fetcher import get_multi_season_results
from label_improvement_test import classify_prediction, label_baseline


def collect_skip_rows(division: str, eval_season: int) -> list[dict]:
    """walk-forward でスキップ層の試合を抽出、特徴量を付与"""
    all_results = get_multi_season_results([2024, 2025], division)
    res = run_walk_forward(
        all_results, eval_season=eval_season,
        predictor="integrated",
        weights=BASELINE_WEIGHTS, params=BASELINE_PARAMS,
    )
    preds = res["predictions"]

    rows = []
    for p in preds:
        idx = p["idx"]
        c = classify_prediction(p["prob_home"], p["prob_draw"], p["prob_away"])
        label = label_baseline(c)
        if label != "スキップ":
            continue  # 対象外

        # 特徴量
        states = rebuild_states(all_results, idx)
        hs = states.get(p["home"])
        as_ = states.get(p["away"])
        elo = build_elo(all_results, idx)
        h_elo, a_elo = elo.score_pair(p["home"], p["away"])

        # ランキング
        ranked = sorted(states, key=lambda t: (-states[t].points, -states[t].goal_diff))
        rank = {t: i+1 for i, t in enumerate(ranked)}
        h_rank = rank.get(p["home"], 99)
        a_rank = rank.get(p["away"], 99)

        # 直近5戦
        h_form = hs.get_form(5) if hs else []
        a_form = as_.get_form(5) if as_ else []
        h_w = sum(1 for r in h_form if r == "W")
        a_w = sum(1 for r in a_form if r == "W")
        h_l = sum(1 for r in h_form if r == "L")
        a_l = sum(1 for r in a_form if r == "L")

        # 得失点差/試合
        h_gd_pg = (hs.goal_diff / max(hs.games, 1)) if hs and hs.games else 0
        a_gd_pg = (as_.goal_diff / max(as_.games, 1)) if as_ and as_.games else 0

        rows.append({
            "division": division,
            "date": p["date"],
            "home": p["home"], "away": p["away"],
            "actual": p["actual"], "pred_winner": p["predicted"],
            "prob_h": p["prob_home"], "prob_d": p["prob_draw"], "prob_a": p["prob_away"],
            "diff": c["diff"],
            "max_prob": c["max_prob"],
            "h_rank": h_rank, "a_rank": a_rank,
            "rank_diff": a_rank - h_rank,  # + なら ホームが上位
            "elo_h": h_elo, "elo_a": a_elo,
            "elo_gap": h_elo - a_elo,  # + ならホーム有利
            "h_form_w": h_w, "a_form_w": a_w,
            "h_form_l": h_l, "a_form_l": a_l,
            "form_w_diff": h_w - a_w,
            "h_gd_pg": h_gd_pg, "a_gd_pg": a_gd_pg,
            "gd_diff": h_gd_pg - a_gd_pg,
            "correct": p["predicted"] == p["actual"],
        })
    return rows


def bucket_accuracy(rows, feature, buckets):
    """各バケットの正答率を返す"""
    out = []
    for name, lo, hi in buckets:
        sub = [r for r in rows if lo <= r[feature] < hi]
        if not sub:
            out.append({"bucket": name, "n": 0, "hit": 0, "acc": None})
            continue
        hit = sum(1 for r in sub if r["correct"])
        out.append({
            "bucket": name, "n": len(sub), "hit": hit,
            "acc": hit / len(sub),
        })
    return out


def print_buckets(title, buckets):
    print(f"\n  {title}")
    print(f"    {'bucket':<20} {'n':>4} {'hit':>4} {'acc':>7}")
    for b in buckets:
        acc = f"{b['acc']*100:>5.1f}%" if b["acc"] is not None else "  —  "
        print(f"    {b['bucket']:<20} {b['n']:>4} {b['hit']:>4} {acc}")


def find_promising_rules(rows: list[dict]) -> list[dict]:
    """複数特徴量の組み合わせで正答率 >= 45% かつ n>=10 のルールを列挙"""
    promising = []

    # 各特徴量の「条件候補」
    conditions = {
        "rank_diff_gt_10":    ("h_rank < a_rank and a_rank - h_rank >= 10", lambda r: r["rank_diff"] >= 10),
        "rank_diff_lt_neg10": ("rank_diff <= -10 (アウェー上位)", lambda r: r["rank_diff"] <= -10),
        "elo_gap_gt_0.15":    ("elo_gap >= 0.15 (ホーム実力上)", lambda r: r["elo_gap"] >= 0.15),
        "elo_gap_lt_neg0.15": ("elo_gap <= -0.15 (アウェー実力上)", lambda r: r["elo_gap"] <= -0.15),
        "h_form_w>=4":        ("h_form_w >= 4", lambda r: r["h_form_w"] >= 4),
        "a_form_w>=4":        ("a_form_w >= 4", lambda r: r["a_form_w"] >= 4),
        "form_w_diff>=3":     ("form_w_diff >= 3 (ホーム好調差)", lambda r: r["form_w_diff"] >= 3),
        "form_w_diff<=-3":    ("form_w_diff <= -3 (アウェー好調差)", lambda r: r["form_w_diff"] <= -3),
        "gd_diff>=0.5":       ("gd_diff >= 0.5 (ホーム得失点強)", lambda r: r["gd_diff"] >= 0.5),
        "gd_diff<=-0.5":      ("gd_diff <= -0.5 (アウェー得失点強)", lambda r: r["gd_diff"] <= -0.5),
        "h_rank_top3":        ("h_rank <= 3", lambda r: r["h_rank"] <= 3),
        "a_rank_top3":        ("a_rank <= 3", lambda r: r["a_rank"] <= 3),
        "h_rank_bottom3":     ("h_rank >= 18", lambda r: r["h_rank"] >= 18),
        "a_rank_bottom3":     ("a_rank >= 18", lambda r: r["a_rank"] >= 18),
    }

    # 単一条件
    print("\n\n  === 単一条件でのスクリーニング ===")
    print(f"    {'condition':<35} {'n':>4} {'hit':>4} {'acc':>7}  {'argmax_home_acc':>16}")
    for key, (desc, fn) in conditions.items():
        sub = [r for r in rows if fn(r)]
        if len(sub) < 10:
            continue
        hit = sum(1 for r in sub if r["correct"])
        # 第一推奨 (argmax) がホームだった試合の的中率も計算
        home_preds = [r for r in sub if r["pred_winner"] == "home"]
        home_hits = sum(1 for r in home_preds if r["correct"])
        home_acc = home_hits / len(home_preds) if home_preds else None
        acc = hit / len(sub)
        print(f"    {key:<35} {len(sub):>4} {hit:>4} {acc*100:>5.1f}%"
              f"  {(home_acc*100 if home_acc is not None else 0):>5.1f}% (n={len(home_preds)})")
        if acc >= 0.45:
            promising.append({
                "condition": desc, "n": len(sub), "hit": hit, "acc": acc,
            })

    # 二重条件 (鉄板ルール探索)
    print("\n\n  === 二重条件 (acc >= 0.45, n >= 10) ===")
    keys = list(conditions.keys())
    print(f"    {'cond_A':<32} {'cond_B':<32} {'n':>4} {'hit':>4} {'acc':>7}")
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            ka, kb = keys[i], keys[j]
            fa, fb = conditions[ka][1], conditions[kb][1]
            sub = [r for r in rows if fa(r) and fb(r)]
            if len(sub) < 10:
                continue
            hit = sum(1 for r in sub if r["correct"])
            acc = hit / len(sub)
            if acc >= 0.45:
                print(f"    {ka:<32} {kb:<32} {len(sub):>4} {hit:>4} {acc*100:>5.1f}%")
                promising.append({
                    "condition_A": ka, "condition_B": kb,
                    "n": len(sub), "hit": hit, "acc": acc,
                })

    return promising


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", default="2025", help="comma-separated")
    parser.add_argument("--save", default="backtest_results/skip_subgroup_analysis.json")
    args = parser.parse_args()

    eval_seasons = [int(s) for s in args.seasons.split(",")]
    rows = []
    for div in ["j1", "j2"]:
        for sy in eval_seasons:
            rows.extend(collect_skip_rows(div, sy))

    if not rows:
        print("No skip-layer matches found.")
        return

    print(f"\n{'='*60}")
    print(f"  スキップ層 サブグループ分析 (n={len(rows)})")
    print(f"{'='*60}")
    hits = sum(1 for r in rows if r["correct"])
    print(f"\n  全体正答率: {hits}/{len(rows)} = {hits/len(rows)*100:.1f}%")
    print(f"  (ランダム 33.3%, baseline スキップ 30% 前後が過去実績)")

    # draw 実割合
    draws = sum(1 for r in rows if r["actual"] == "draw")
    home_w = sum(1 for r in rows if r["actual"] == "home")
    away_w = sum(1 for r in rows if r["actual"] == "away")
    print(f"\n  実結果分布: home {home_w}勝 / draw {draws} / away {away_w}勝")

    # pred_winner 別の内訳
    print(f"\n  pred_winner 別の正答率:")
    for pw in ["home", "draw", "away"]:
        sub = [r for r in rows if r["pred_winner"] == pw]
        if not sub:
            continue
        h = sum(1 for r in sub if r["correct"])
        print(f"    pred={pw:<6}: n={len(sub):>3}  hit={h:>3}  acc={h/len(sub)*100:>5.1f}%")

    # バケット別分析
    print_buckets("rank_diff (+: ホーム上位)", bucket_accuracy(rows, "rank_diff", [
        ("rank_diff <=-10", -99, -10),
        ("-10 < rank_diff <=-3", -10, -3),
        ("-3 < rank_diff < 3", -3, 3),
        ("3 <= rank_diff < 10", 3, 10),
        ("rank_diff >= 10", 10, 99),
    ]))

    print_buckets("elo_gap (+: ホーム実力上)", bucket_accuracy(rows, "elo_gap", [
        ("<= -0.2",     -9, -0.2),
        ("-0.2〜-0.05", -0.2, -0.05),
        ("-0.05〜0.05", -0.05, 0.05),
        ("0.05〜0.2",   0.05, 0.2),
        (">= 0.2",      0.2, 9),
    ]))

    print_buckets("form_w_diff", bucket_accuracy(rows, "form_w_diff", [
        ("<= -3", -99, -3),
        ("-2〜-1", -2, 0),
        ("0",      0, 1),
        ("1〜2",   1, 3),
        (">= 3",   3, 99),
    ]))

    print_buckets("gd_diff (得失点差/試合)", bucket_accuracy(rows, "gd_diff", [
        ("<= -0.8", -99, -0.8),
        ("-0.8〜-0.3", -0.8, -0.3),
        ("-0.3〜0.3",  -0.3, 0.3),
        ("0.3〜0.8",   0.3, 0.8),
        (">= 0.8",     0.8, 99),
    ]))

    promising = find_promising_rules(rows)

    out = Path(args.save)
    out.parent.mkdir(exist_ok=True, parents=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "n_skip_total": len(rows),
            "overall_accuracy": hits / len(rows),
            "seasons": eval_seasons,
            "promising_rules": promising,
            "rows": rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
