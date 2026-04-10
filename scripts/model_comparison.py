"""
scripts/model_comparison.py - 全モデル統合比較

比較対象:
- v7 (primary)
- v8.1 (shadow)
- v8.1 + Temperature Scaling
- v8.1 + Isotonic Regression
- xG-Skellam
- xG-Skellam + Calibration
- elo_only
- 軽量アンサンブル (v7 + v8.1 + Skellam)

評価: val=2025 と 2026 holdout
指標: accuracy, F1 macro, log loss, Brier, ECE, draw precision/recall/F1
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import predict_logic
from data_fetcher import get_multi_season_results
from scripts.backtest_runner import (
    run_walk_forward, rebuild_states, compute_ranks, build_elo, compute_metrics
)
from scripts.skellam_model import predict_skellam, predict_skellam_dynamic, compute_dynamic_draw_boost
from scripts.calibration import (
    TemperatureScaler, IsotonicCalibrator, predictions_to_arrays, compute_ece
)


LABELS = ["away", "draw", "home"]


def _metrics_from_probs(probs: np.ndarray, y_idx: np.ndarray, class_names=LABELS) -> dict:
    """確率配列から評価指標を算出"""
    from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix

    n = len(y_idx)
    if n == 0:
        return {}
    preds = np.argmax(probs, axis=1)
    y_true_labels = [class_names[i] for i in y_idx]
    y_pred_labels = [class_names[i] for i in preds]

    acc = accuracy_score(y_idx, preds)
    f1 = f1_score(y_idx, preds, average="macro", zero_division=0)
    pm = np.clip(probs, 0.01, 0.99)
    pm = pm / pm.sum(axis=1, keepdims=True)
    ll = log_loss(y_idx, pm, labels=[0, 1, 2])
    y_oh = np.zeros_like(probs)
    y_oh[np.arange(n), y_idx] = 1
    brier = float(np.mean(np.sum((pm - y_oh) ** 2, axis=1)))
    ece = compute_ece(pm, y_idx)
    cm = confusion_matrix(y_idx, preds, labels=[0, 1, 2])

    # クラス別
    class_metrics = {}
    for i, c in enumerate(class_names):
        tp = int(cm[i][i])
        fp = int(sum(cm[j][i] for j in range(3)) - tp)
        fn = int(sum(cm[i]) - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1c = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        class_metrics[c] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1c, 4)}

    pred_dist = {c: int((preds == i).sum()) for i, c in enumerate(class_names)}

    return {
        "n": n,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
        "log_loss": round(ll, 4),
        "brier": round(brier, 4),
        "ece": round(ece, 4),
        "class_metrics": class_metrics,
        "pred_dist": pred_dist,
    }


def _run_predictor_with_weights(all_results, eval_season, model_weights, logit_params):
    """MODEL_WEIGHTSと_3LOGIT_PARAMSを一時差し替えて integrated を実行"""
    orig_w = dict(predict_logic.MODEL_WEIGHTS)
    orig_p = dict(predict_logic._3LOGIT_PARAMS)
    try:
        predict_logic.MODEL_WEIGHTS.clear()
        predict_logic.MODEL_WEIGHTS.update(model_weights)
        predict_logic._3LOGIT_PARAMS.clear()
        predict_logic._3LOGIT_PARAMS.update(logit_params)
        res = run_walk_forward(all_results, eval_season=eval_season, predictor="integrated")
    finally:
        predict_logic.MODEL_WEIGHTS.clear()
        predict_logic.MODEL_WEIGHTS.update(orig_w)
        predict_logic._3LOGIT_PARAMS.clear()
        predict_logic._3LOGIT_PARAMS.update(orig_p)
    return res


def _run_skellam_walk_forward(
    all_results: list[dict], eval_season: int,
    home_advantage: float = 0.25,
    rho: float = -0.08,
    draw_boost: float = 0.0,
    use_elo: bool = False,
) -> dict:
    """Skellam モデルの walk-forward 評価"""
    predictions = []
    for idx, match in enumerate(all_results):
        if match.get("season") != eval_season:
            continue
        actual = match.get("winner")
        if not actual:
            continue
        home = match["home"]
        away = match["away"]

        states = rebuild_states(all_results, idx)
        hs = states.get(home)
        as_ = states.get(away)
        if not hs or not as_:
            continue

        ranks = compute_ranks(states)
        h_stats = hs.to_stats_dict(ranks.get(home, 99))
        a_stats = as_.to_stats_dict(ranks.get(away, 99))

        # ELO取得 (オプション)
        elo_h = elo_a = None
        if use_elo:
            elo = build_elo(all_results, idx, k=32.0, home_bonus=50.0)
            elo_h, elo_a = elo.score_pair(home, away)

        sp = predict_skellam(
            h_stats, a_stats,
            home_advantage=home_advantage,
            rho=rho,
            draw_boost=draw_boost,
            elo_home_score=elo_h,
            elo_away_score=elo_a,
        )
        predictions.append({
            "idx": idx,
            "date": match.get("date", ""),
            "section": match.get("section", 0),
            "home": home,
            "away": away,
            "actual": actual,
            "prob_home": sp["home_win_prob"] / 100,
            "prob_draw": sp["draw_prob"] / 100,
            "prob_away": sp["away_win_prob"] / 100,
            "predicted": max([("home", sp["home_win_prob"]), ("draw", sp["draw_prob"]),
                              ("away", sp["away_win_prob"])], key=lambda x: x[1])[0],
        })

    if not predictions:
        return {"predictions": [], "metrics": {}}

    probs, y_idx = predictions_to_arrays(predictions)
    return {"predictions": predictions, "metrics": _metrics_from_probs(probs, y_idx)}


def _run_skellam_dynamic_walk_forward(all_results: list[dict], eval_season: int) -> dict:
    """動的boost付きSkellamの walk-forward 評価"""
    predictions = []
    for idx, match in enumerate(all_results):
        if match.get("season") != eval_season:
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

        elo = build_elo(all_results, idx, k=32.0, home_bonus=50.0)
        elo_h, elo_a = elo.score_pair(home, away)

        sp = predict_skellam_dynamic(
            h_stats, a_stats,
            elo_home_score=elo_h, elo_away_score=elo_a,
        )
        predictions.append({
            "idx": idx, "date": match.get("date", ""),
            "section": match.get("section", 0),
            "home": home, "away": away, "actual": actual,
            "prob_home": sp["home_win_prob"] / 100,
            "prob_draw": sp["draw_prob"] / 100,
            "prob_away": sp["away_win_prob"] / 100,
            "predicted": max([("home", sp["home_win_prob"]), ("draw", sp["draw_prob"]),
                              ("away", sp["away_win_prob"])], key=lambda x: x[1])[0],
            "dynamic_boost": sp.get("dynamic_boost", 0.0),
        })

    if not predictions:
        return {"predictions": [], "metrics": {}}
    probs, y_idx = predictions_to_arrays(predictions)
    return {"predictions": predictions, "metrics": _metrics_from_probs(probs, y_idx)}


def _preds_to_probs_labels(res: dict) -> tuple[np.ndarray, np.ndarray]:
    preds = res.get("predictions", [])
    if not preds:
        return np.array([]), np.array([])
    return predictions_to_arrays(preds)


def run_full_comparison() -> dict:
    """全モデルを評価して結果を返す"""
    print("Fetching data...")
    all_results = get_multi_season_results([2024, 2025], "j1")
    print(f"Total: {len(all_results)}")

    models = {}

    # --- v7 (primary) ---
    print("\n[1/7] v7 (primary)...")
    v7_w = dict(predict_logic.MODEL_WEIGHTS)
    v7_p = dict(predict_logic._3LOGIT_PARAMS)
    for season in [2025, 2026]:
        res = _run_predictor_with_weights(all_results, season, v7_w, v7_p)
        probs, y_idx = _preds_to_probs_labels(res)
        if len(probs):
            res["metrics"] = _metrics_from_probs(probs, y_idx)
        models.setdefault("v7", {})[season] = res

    # --- v8.1 (shadow) ---
    print("[2/7] v8.1 (shadow)...")
    for season in [2025, 2026]:
        res = _run_predictor_with_weights(
            all_results, season,
            predict_logic.V8_1_MODEL_WEIGHTS,
            predict_logic.V8_1_3LOGIT_PARAMS,
        )
        probs, y_idx = _preds_to_probs_labels(res)
        if len(probs):
            res["metrics"] = _metrics_from_probs(probs, y_idx)
        models.setdefault("v8.1", {})[season] = res

    # --- Calibration for v8.1 (train on 2024) ---
    print("[3/7] v8.1 calibration (trained on 2024)...")
    res_v8_train = _run_predictor_with_weights(
        all_results, 2024,
        predict_logic.V8_1_MODEL_WEIGHTS,
        predict_logic.V8_1_3LOGIT_PARAMS,
    )
    probs_train, y_train = _preds_to_probs_labels(res_v8_train)

    # Temperature scaling
    temp_cal = TemperatureScaler().fit(probs_train, y_train)
    print(f"   Temperature T = {temp_cal.T:.3f}")
    # Isotonic
    iso_cal = IsotonicCalibrator().fit(probs_train, y_train)

    # --- v8.1 + Temperature ---
    print("[4/7] v8.1 + Temperature...")
    for season in [2025, 2026]:
        base_res = models["v8.1"][season]
        probs, y_idx = _preds_to_probs_labels(base_res)
        if len(probs):
            calibrated = temp_cal.transform(probs)
            m = _metrics_from_probs(calibrated, y_idx)
        else:
            m = {}
        models.setdefault("v8.1+temp", {})[season] = {"metrics": m, "predictions": base_res["predictions"]}

    # --- v8.1 + Isotonic ---
    print("[5/7] v8.1 + Isotonic...")
    for season in [2025, 2026]:
        base_res = models["v8.1"][season]
        probs, y_idx = _preds_to_probs_labels(base_res)
        if len(probs):
            calibrated = iso_cal.transform(probs)
            m = _metrics_from_probs(calibrated, y_idx)
        else:
            m = {}
        models.setdefault("v8.1+iso", {})[season] = {"metrics": m, "predictions": base_res["predictions"]}

    # --- xG-Skellam (baseline, rho=-0.08) ---
    print("[6/7] xG-Skellam...")
    for season in [2025, 2026]:
        res = _run_skellam_walk_forward(all_results, season)
        models.setdefault("skellam", {})[season] = res

    # --- Skellam+ with draw boost ---
    print("      Skellam+draw_boost...")
    for season in [2025, 2026]:
        res = _run_skellam_walk_forward(
            all_results, season,
            draw_boost=0.10, use_elo=True,
        )
        models.setdefault("skellam+boost", {})[season] = res

    # --- Skellam with stronger DC correction ---
    print("      Skellam+DC強化...")
    for season in [2025, 2026]:
        res = _run_skellam_walk_forward(
            all_results, season,
            rho=-0.15, draw_boost=0.08, use_elo=True,
        )
        models.setdefault("skellam+dc", {})[season] = res

    # --- Skellam dynamic (動的boost) ---
    print("      Skellam dynamic...")
    for season in [2025, 2026]:
        res = _run_skellam_dynamic_walk_forward(all_results, season)
        models.setdefault("skellam_dyn", {})[season] = res

    # --- elo_only ---
    print("[7/7] elo_only...")
    for season in [2025, 2026]:
        res = run_walk_forward(all_results, eval_season=season, predictor="elo_only")
        # metricsを再計算 (ECE追加のため)
        probs, y_idx = _preds_to_probs_labels(res)
        if len(probs):
            res["metrics"] = _metrics_from_probs(probs, y_idx)
        models.setdefault("elo_only", {})[season] = res

    # --- Hybrid v9: Model selection with dynamic boost Skellam ---
    # ルール:
    #   - v7のdraw警戒時 → v7採用
    #   - Skellam dynamicが高確信 (max_prob >= 50%) → Skellam採用
    #   - それ以外 → v7とSkellamの重み付き平均 (0.5/0.5)
    print("\n[hybrid v9] draw警戒→v7, 高確信→Skellam_dyn, else→weighted")
    for season in [2025, 2026]:
        v7_preds = models["v7"][season].get("predictions", [])
        sk_preds = models["skellam_dyn"][season].get("predictions", [])
        if not v7_preds or not sk_preds:
            continue
        v7_map = {p["idx"]: p for p in v7_preds}
        sk_map = {p["idx"]: p for p in sk_preds}
        common = sorted(set(v7_map) & set(sk_map))

        hyb_probs = []
        actuals = []
        selection_log = {"v7": 0, "skellam": 0, "weighted": 0}
        for idx in common:
            v7p = v7_map[idx]
            skp = sk_map[idx]
            v7_probs = np.array([v7p["prob_away"], v7p["prob_draw"], v7p["prob_home"]])
            sk_probs = np.array([skp["prob_away"], skp["prob_draw"], skp["prob_home"]])

            # v7のdraw警戒判定
            v7_draw_alert = (v7p["prob_draw"] >= 0.25 and
                             abs(v7p["prob_home"] - v7p["prob_away"]) < 0.10)
            # Skellamの高確信判定 (非draw argmax)
            sk_max = max(sk_probs)
            sk_argmax = np.argmax(sk_probs)
            sk_high_conf_nondraw = (sk_max >= 0.50 and sk_argmax != 1)

            if v7_draw_alert:
                chosen = v7_probs
                selection_log["v7"] += 1
            elif sk_high_conf_nondraw:
                chosen = sk_probs
                selection_log["skellam"] += 1
            else:
                chosen = (v7_probs + sk_probs) / 2.0
                chosen = chosen / chosen.sum()
                selection_log["weighted"] += 1
            hyb_probs.append(chosen)
            actuals.append({"away": 0, "draw": 1, "home": 2}[v7p["actual"]])

        hyb_arr = np.array(hyb_probs)
        y_arr = np.array(actuals)
        m = _metrics_from_probs(hyb_arr, y_arr)
        m["selection_log"] = selection_log
        models.setdefault("hybrid_v9", {})[season] = {"metrics": m}

    # --- Selection Ensemble: draw警戒時はv7, それ以外はSkellam+boost ---
    print("\n[selection] draw警戒→v7, それ以外→Skellam+...")
    for season in [2025, 2026]:
        v7_preds = models["v7"][season].get("predictions", [])
        sk_preds = models["skellam+boost"][season].get("predictions", [])
        if not v7_preds or not sk_preds:
            continue
        v7_map = {p["idx"]: p for p in v7_preds}
        sk_map = {p["idx"]: p for p in sk_preds}
        common = sorted(set(v7_map) & set(sk_map))

        sel_probs = []
        actuals = []
        for idx in common:
            v7p = v7_map[idx]
            # draw警戒判定: draw_prob>=25% かつ closeness高い (home/away接近)
            draw_alert = v7p["prob_draw"] >= 0.25 and abs(v7p["prob_home"] - v7p["prob_away"]) < 0.10
            if draw_alert:
                # v7採用
                probs = [v7p["prob_away"], v7p["prob_draw"], v7p["prob_home"]]
            else:
                # Skellam+採用
                skp = sk_map[idx]
                probs = [skp["prob_away"], skp["prob_draw"], skp["prob_home"]]
            sel_probs.append(probs)
            actuals.append({"away": 0, "draw": 1, "home": 2}[v7p["actual"]])

        sel_arr = np.array(sel_probs)
        y_arr = np.array(actuals)
        m = _metrics_from_probs(sel_arr, y_arr)
        models.setdefault("selection", {})[season] = {"metrics": m}

    # --- 軽量アンサンブル: v7 + v8.1+temp + skellam の平均 ---
    print("\n[ensemble] v7 + v8.1+temp + skellam 平均...")
    for season in [2025, 2026]:
        v7_preds = models["v7"][season].get("predictions", [])
        v81t_preds = models["v8.1+temp"][season].get("predictions", [])
        sk_preds = models["skellam"][season].get("predictions", [])
        if not v7_preds or not v81t_preds or not sk_preds:
            continue
        # match_idxで揃える
        v7_map = {p["idx"]: p for p in v7_preds}
        v81_map = {p["idx"]: p for p in v81t_preds}
        sk_map = {p["idx"]: p for p in sk_preds}
        common = set(v7_map) & set(v81_map) & set(sk_map)
        if not common:
            continue
        sorted_idx = sorted(common)

        # v8.1+tempのtemperature適用後の確率を再計算
        v81_probs, _ = _preds_to_probs_labels(models["v8.1"][season])
        v81_temp_probs = temp_cal.transform(v81_probs)
        # idx→probsマップ
        v81_probs_map = {}
        for i, p in enumerate(models["v8.1"][season]["predictions"]):
            v81_probs_map[p["idx"]] = v81_temp_probs[i]

        ens_probs = []
        actuals = []
        for idx in sorted_idx:
            p7 = np.array([v7_map[idx]["prob_away"], v7_map[idx]["prob_draw"], v7_map[idx]["prob_home"]])
            p81 = v81_probs_map[idx]
            ps = np.array([sk_map[idx]["prob_away"], sk_map[idx]["prob_draw"], sk_map[idx]["prob_home"]])
            avg = (p7 + p81 + ps) / 3
            avg = avg / avg.sum()
            ens_probs.append(avg)
            actuals.append({"away": 0, "draw": 1, "home": 2}[v7_map[idx]["actual"]])

        ens_arr = np.array(ens_probs)
        y_arr = np.array(actuals)
        m = _metrics_from_probs(ens_arr, y_arr)
        models.setdefault("ensemble", {})[season] = {"metrics": m}

    return models


def print_report(models: dict) -> None:
    print("\n" + "=" * 90)
    print("  MODEL COMPARISON REPORT")
    print("=" * 90)

    for season in [2025, 2026]:
        print(f"\n--- val/holdout = {season} ---")
        print(f"{'Model':<15} {'n':>4} {'Acc':>7} {'F1':>7} {'LogL':>7} {'Brier':>7} {'ECE':>6} {'Draw#':>6} {'DrawF1':>7}")
        print("-" * 80)
        for name, data in models.items():
            res = data.get(season, {})
            m = res.get("metrics", {})
            if not m:
                continue
            n = m.get("n") or m.get("n_samples", 0)
            acc = m.get("accuracy", 0)
            f1 = m.get("f1_macro", 0)
            ll = m.get("log_loss", 0)
            br = m.get("brier", 0)
            ece = m.get("ece", 0)
            dp = m.get("pred_dist", {}).get("draw", 0)
            df1 = m.get("class_metrics", {}).get("draw", {}).get("f1", 0)
            print(f"{name:<15} {n:>4} {acc:>7.4f} {f1:>7.4f} {ll:>7.4f} {br:>7.4f} {ece:>6.4f} {dp:>6} {df1:>7.4f}")


if __name__ == "__main__":
    models = run_full_comparison()
    print_report(models)

    # 保存
    out_dir = Path(__file__).parent.parent / "backtest_results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "model_comparison_latest.json"
    summary = {}
    for name, data in models.items():
        summary[name] = {
            str(season): {"metrics": data[season].get("metrics", {})}
            for season in data
        }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved: {out_path}")
