"""
scripts/backtest_runner.py - Jリーグ予測モデル バックテスト基盤

時系列を厳守した walk-forward validation でモデルを評価する。
未来情報リークを防ぐため、各試合の予測には「その試合前」のデータのみを使用。

Usage:
    python scripts/backtest_runner.py                     # ベースライン評価
    python scripts/backtest_runner.py --optimize           # Optuna最適化
    python scripts/backtest_runner.py --experiment-id test1 --notes "dead param removal"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# scripts/ から親ディレクトリをインポート可能にする
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).parent.parent
LOG_PATH = ROOT / "experiment_logs.csv"
RESULTS_DIR = ROOT / "backtest_results"
RESULTS_DIR.mkdir(exist_ok=True)

# ────────────────────────────────────────────────────────
# 1. 過去試合データの取得と順位表再構築
# ────────────────────────────────────────────────────────

def fetch_past_results(division: str = "j1") -> list[dict]:
    """jleague.jpから過去試合結果を取得"""
    from data_fetcher import get_past_results
    results = get_past_results(division)
    # 日付順にソート
    results.sort(key=lambda x: (x.get("date", ""), x.get("time", "")))
    return results


@dataclass
class TeamState:
    """ある時点でのチーム累積状態"""
    games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    points: int = 0
    recent_results: list = field(default_factory=list)  # ['W','D','L',...]

    @property
    def goal_diff(self) -> int:
        return self.goals_for - self.goals_against

    @property
    def attack_rate(self) -> float:
        return self.goals_for / max(self.games, 1)

    @property
    def defense_rate(self) -> float:
        return self.goals_against / max(self.games, 1)

    def to_stats_dict(self, rank: int = 0) -> dict:
        """predict_logic.pyのscore_*関数が期待するdict形式に変換"""
        return {
            "順位": str(rank),
            "勝点": str(self.points),
            "試合": str(self.games),
            "勝": str(self.wins),
            "分": str(self.draws),
            "負": str(self.losses),
            "得点": str(self.goals_for),
            "失点": str(self.goals_against),
            "得失点差": f"{self.goal_diff:+d}" if self.goal_diff >= 0 else str(self.goal_diff),
        }

    def get_form(self, n: int = 5) -> list[str]:
        """直近n試合のフォーム"""
        return self.recent_results[-n:] if self.recent_results else []


def rebuild_standings_at_match(
    all_results: list[dict],
    match_index: int,
) -> dict[str, TeamState]:
    """
    match_index番目の試合「より前」の結果のみでチーム状態を再構築。
    未来情報リークを防ぐ核心ロジック。
    """
    states: dict[str, TeamState] = defaultdict(TeamState)

    for i in range(match_index):
        r = all_results[i]
        home = r["home"]
        away = r["away"]
        h_score = int(r.get("home_score", 0))
        a_score = int(r.get("away_score", 0))

        for team, gf, ga, is_home in [
            (home, h_score, a_score, True),
            (away, a_score, h_score, False),
        ]:
            s = states[team]
            s.games += 1
            s.goals_for += gf
            s.goals_against += ga

            if gf > ga:
                s.wins += 1
                s.points += 3
                s.recent_results.append("W")
            elif gf == ga:
                s.draws += 1
                s.points += 1
                s.recent_results.append("D")
            else:
                s.losses += 1
                s.recent_results.append("L")

    # 順位計算 (勝点 → 得失点差 → 得点の順)
    ranked = sorted(
        states.items(),
        key=lambda x: (-x[1].points, -x[1].goal_diff, -x[1].goals_for),
    )
    for rank, (team, _) in enumerate(ranked, 1):
        pass  # rank は to_stats_dict で使用

    return states


def compute_ranks(states: dict[str, TeamState]) -> dict[str, int]:
    """チーム状態から順位を計算"""
    ranked = sorted(
        states.keys(),
        key=lambda t: (-states[t].points, -states[t].goal_diff, -states[t].goals_for),
    )
    return {team: rank + 1 for rank, team in enumerate(ranked)}


# ────────────────────────────────────────────────────────
# 1b. ELO レーティングシステム
# ────────────────────────────────────────────────────────

class EloSystem:
    """試合結果から動的に更新するELOレーティング"""

    def __init__(self, k: float = 32.0, initial: float = 1500.0, home_bonus: float = 50.0):
        self.k = k
        self.initial = initial
        self.home_bonus = home_bonus
        self.ratings: dict[str, float] = {}

    def get(self, team: str) -> float:
        return self.ratings.get(team, self.initial)

    def expected(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def update(self, home: str, away: str, winner: str) -> None:
        r_home = self.get(home) + self.home_bonus
        r_away = self.get(away)
        e_home = self.expected(r_home, r_away)

        if winner == "home":
            s_home, s_away = 1.0, 0.0
        elif winner == "away":
            s_home, s_away = 0.0, 1.0
        else:
            s_home, s_away = 0.5, 0.5

        self.ratings[home] = self.get(home) + self.k * (s_home - e_home)
        self.ratings[away] = self.get(away) + self.k * (s_away - (1.0 - e_home))

    def score_pair(self, home: str, away: str) -> tuple[float, float]:
        """ELOを0-1スコアに変換 (対戦相手との相対値)"""
        r_home = self.get(home) + self.home_bonus
        r_away = self.get(away)
        e_home = self.expected(r_home, r_away)
        return e_home, 1.0 - e_home


def build_elo_at_match(
    all_results: list[dict],
    match_index: int,
    k: float = 32.0,
    home_bonus: float = 50.0,
) -> EloSystem:
    """match_index番目の試合より前の結果でELOを構築"""
    elo = EloSystem(k=k, home_bonus=home_bonus)
    for i in range(match_index):
        r = all_results[i]
        elo.update(r["home"], r["away"], r["winner"])
    return elo


# ────────────────────────────────────────────────────────
# 2. 予測エンジン (統計モデルのみ - 再現可能)
# ────────────────────────────────────────────────────────

def predict_match_statistical(
    home_team: str,
    away_team: str,
    home_stats: dict,
    away_stats: dict,
    home_form: list[str],
    away_form: list[str],
    weights: dict[str, float],
    params: dict[str, Any] | None = None,
    elo: EloSystem | None = None,
) -> dict[str, float]:
    """
    重み付き線形モデルで予測確率を返す。
    Gemini不使用 = 完全再現可能。

    Returns: {"home": float, "draw": float, "away": float}  # 確率 [0,1]
    """
    from predict_logic import (
        score_team_strength,
        score_attack_rate,
        score_defense_rate,
        score_recent_form,
        score_home_advantage,
        score_capital_power,
        MODEL_WEIGHTS,
    )

    params = params or {}
    sigmoid_k = params.get("sigmoid_k", 3.0)
    sigmoid_m = params.get("sigmoid_m", 0.30)
    base_home = params.get("base_home", 0.40)
    base_draw = params.get("base_draw", 0.25)
    base_away = params.get("base_away", 0.35)
    form_n = params.get("form_n", 5)
    draw_closeness_boost = params.get("draw_closeness_boost", 0.0)

    # スコア計算
    h_str, a_str = score_team_strength(home_stats, away_stats)
    h_atk, a_atk = score_attack_rate(home_stats, away_stats)
    h_def, a_def = score_defense_rate(home_stats, away_stats)
    h_form = score_recent_form(home_form[-form_n:])
    a_form = score_recent_form(away_form[-form_n:])
    h_home = score_home_advantage()
    a_away = 1.0 - h_home
    h_cap, a_cap = score_capital_power(home_team, away_team)

    # 有効パラメータのみで加重合計
    scores: dict[str, tuple[float, float]] = {
        "team_strength": (h_str, a_str),
        "attack_rate": (h_atk, a_atk),
        "defense_rate": (h_def, a_def),
        "recent_form": (h_form, a_form),
        "home_advantage": (h_home, a_away),
        "capital_power": (h_cap, a_cap),
    }

    # ELOスコア (オプション)
    if elo and weights.get("elo", 0) > 0:
        h_elo, a_elo = elo.score_pair(home_team, away_team)
        scores["elo"] = (h_elo, a_elo)

    raw_adv = 0.0
    total_weight = 0.0
    for name, (h_s, a_s) in scores.items():
        w = weights.get(name, 0.0)
        raw_adv += (h_s - a_s) * w
        total_weight += w

    # 重み正規化 (合計が1でない場合)
    if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
        raw_adv /= total_weight

    # シグモイド変換
    shift = math.tanh(raw_adv * sigmoid_k) * sigmoid_m

    h = max(0.05, min(0.90, base_home + shift))
    a = max(0.05, min(0.90, base_away - shift))

    # 引き分け補正: 実力接近度に基づくdraw確率のブースト
    closeness = max(0, 1.0 - abs(raw_adv) * 2)  # 0〜1, 接近するほど大きい
    draw_boost = draw_closeness_boost * closeness
    d = max(0.05, min(0.50, (1.0 - h - a) + draw_boost))

    # 正規化
    total = h + d + a
    return {
        "home": h / total,
        "draw": d / total,
        "away": a / total,
    }


# ────────────────────────────────────────────────────────
# 3. 評価メトリクス
# ────────────────────────────────────────────────────────

def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    y_prob: list[dict[str, float]],
) -> dict[str, Any]:
    """
    予測結果の評価メトリクスを計算。

    Parameters
    ----------
    y_true : 実際の結果 ["home", "draw", "away", ...]
    y_pred : 予測結果 ["home", "draw", "away", ...]
    y_prob : 予測確率 [{"home": 0.4, "draw": 0.25, "away": 0.35}, ...]
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        log_loss,
        confusion_matrix,
        classification_report,
    )

    n = len(y_true)
    if n == 0:
        return {"error": "no data"}

    labels = ["away", "draw", "home"]

    # accuracy
    acc = accuracy_score(y_true, y_pred)

    # macro F1
    f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    # log loss
    prob_matrix = np.array([[p.get(c, 0.01) for c in labels] for p in y_prob])
    prob_matrix = np.clip(prob_matrix, 0.01, 0.99)
    prob_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)
    ll = log_loss(y_true, prob_matrix, labels=labels)

    # Brier score
    y_true_onehot = np.array([[1 if y == c else 0 for c in labels] for y in y_true])
    brier = float(np.mean(np.sum((prob_matrix - y_true_onehot) ** 2, axis=1)))

    # 混同行列
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    return {
        "n_samples": n,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
        "log_loss": round(ll, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "class_distribution": {
            c: sum(1 for y in y_true if y == c) for c in labels
        },
    }


# ────────────────────────────────────────────────────────
# 4. バックテスト実行
# ────────────────────────────────────────────────────────

def run_backtest(
    results: list[dict],
    weights: dict[str, float],
    params: dict[str, Any] | None = None,
    min_games_before: int = 1,
    val_ratio: float = 0.3,
) -> dict[str, Any]:
    """
    Walk-forward バックテスト。

    Parameters
    ----------
    results : 日付順ソート済み試合結果リスト
    weights : パラメータ重み
    params  : 追加パラメータ (sigmoid_k, sigmoid_m, base_*, form_n)
    min_games_before : 各チーム最低この試合数の履歴が必要
    val_ratio : 後半何割をvalidation setとするか

    Returns
    -------
    {
        "train_metrics": {...},
        "val_metrics": {...},
        "all_metrics": {...},
        "predictions": [...],
    }
    """
    params = params or {}
    predictions = []

    for idx in range(len(results)):
        match = results[idx]
        home = match["home"]
        away = match["away"]
        actual = match.get("winner")

        if not actual:
            continue

        # この試合前のチーム状態を再構築
        states = rebuild_standings_at_match(results, idx)
        ranks = compute_ranks(states)

        h_state = states.get(home)
        a_state = states.get(away)

        # 最低試合数チェック
        if not h_state or h_state.games < min_games_before:
            continue
        if not a_state or a_state.games < min_games_before:
            continue

        h_stats = h_state.to_stats_dict(ranks.get(home, 99))
        a_stats = a_state.to_stats_dict(ranks.get(away, 99))

        form_n = params.get("form_n", 5)
        h_form = h_state.get_form(form_n)
        a_form = a_state.get_form(form_n)

        # ELO構築 (使用する場合)
        elo_obj = None
        if weights.get("elo", 0) > 0:
            elo_k = params.get("elo_k", 32.0)
            elo_hb = params.get("elo_home_bonus", 50.0)
            elo_obj = build_elo_at_match(results, idx, k=elo_k, home_bonus=elo_hb)

        # 予測
        probs = predict_match_statistical(
            home, away, h_stats, a_stats, h_form, a_form,
            weights=weights, params=params, elo=elo_obj,
        )

        # argmax
        pred = max(probs, key=probs.get)

        predictions.append({
            "match_index": idx,
            "date": match.get("date", ""),
            "home": home,
            "away": away,
            "actual": actual,
            "predicted": pred,
            "prob_home": round(probs["home"], 4),
            "prob_draw": round(probs["draw"], 4),
            "prob_away": round(probs["away"], 4),
        })

    if not predictions:
        return {"error": "no evaluable matches", "predictions": []}

    # train / val 分割 (時系列順)
    n = len(predictions)
    split_idx = int(n * (1 - val_ratio))

    train_preds = predictions[:split_idx]
    val_preds = predictions[split_idx:]

    def _extract(preds):
        yt = [p["actual"] for p in preds]
        yp = [p["predicted"] for p in preds]
        yprob = [{"home": p["prob_home"], "draw": p["prob_draw"], "away": p["prob_away"]} for p in preds]
        return yt, yp, yprob

    all_yt, all_yp, all_yprob = _extract(predictions)
    train_yt, train_yp, train_yprob = _extract(train_preds) if train_preds else ([], [], [])
    val_yt, val_yp, val_yprob = _extract(val_preds) if val_preds else ([], [], [])

    return {
        "all_metrics": compute_metrics(all_yt, all_yp, all_yprob),
        "train_metrics": compute_metrics(train_yt, train_yp, train_yprob) if train_preds else {},
        "val_metrics": compute_metrics(val_yt, val_yp, val_yprob) if val_preds else {},
        "n_total": n,
        "n_train": len(train_preds),
        "n_val": len(val_preds),
        "predictions": predictions,
    }


# ────────────────────────────────────────────────────────
# 5. 実験ログ保存
# ────────────────────────────────────────────────────────

LOG_FIELDS = [
    "timestamp", "experiment_id", "phase", "parameter_set", "features_used",
    "val_accuracy", "val_f1_macro", "val_log_loss",
    "holdout_accuracy", "holdout_f1_macro", "holdout_log_loss",
    "train_accuracy", "n_train", "n_val",
    "notes",
]


def save_experiment_log(
    experiment_id: str,
    phase: str,
    weights: dict,
    features: list[str],
    val_metrics: dict,
    holdout_metrics: dict | None = None,
    train_metrics: dict | None = None,
    n_train: int = 0,
    n_val: int = 0,
    notes: str = "",
) -> None:
    """実験結果をCSVに追記"""
    write_header = not LOG_PATH.exists()

    row = {
        "timestamp": datetime.now().isoformat(),
        "experiment_id": experiment_id,
        "phase": phase,
        "parameter_set": json.dumps(weights, ensure_ascii=False),
        "features_used": json.dumps(features, ensure_ascii=False),
        "val_accuracy": val_metrics.get("accuracy", ""),
        "val_f1_macro": val_metrics.get("f1_macro", ""),
        "val_log_loss": val_metrics.get("log_loss", ""),
        "holdout_accuracy": holdout_metrics.get("accuracy", "") if holdout_metrics else "",
        "holdout_f1_macro": holdout_metrics.get("f1_macro", "") if holdout_metrics else "",
        "holdout_log_loss": holdout_metrics.get("log_loss", "") if holdout_metrics else "",
        "train_accuracy": train_metrics.get("accuracy", "") if train_metrics else "",
        "n_train": n_train,
        "n_val": n_val,
        "notes": notes,
    }

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    logger.info("Experiment logged: %s (val_acc=%.4f)", experiment_id, val_metrics.get("accuracy", 0))


# ────────────────────────────────────────────────────────
# 6. Optuna 最適化
# ────────────────────────────────────────────────────────

def optimize_weights(
    results: list[dict],
    n_trials: int = 200,
    patience: int = 20,
    min_improvement: float = 0.001,
    phase: str = "phase2",
) -> dict[str, Any]:
    """
    Optunaで重みとハイパーパラメータを探索。
    停止条件: patience回連続改善なし or n_trials到達 or 改善幅閾値未満。
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    best_val_acc = -1.0
    no_improve_count = 0
    recent_improvements = []

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_val_acc, no_improve_count, recent_improvements

        # 重み探索 (Dirichlet-like: 各要素を独立にサンプルし正規化)
        raw_weights = {
            "team_strength": trial.suggest_float("w_team_strength", 0.01, 0.40),
            "attack_rate": trial.suggest_float("w_attack_rate", 0.01, 0.25),
            "defense_rate": trial.suggest_float("w_defense_rate", 0.01, 0.25),
            "recent_form": trial.suggest_float("w_recent_form", 0.01, 0.40),
            "home_advantage": trial.suggest_float("w_home_advantage", 0.01, 0.25),
            "capital_power": trial.suggest_float("w_capital_power", 0.00, 0.25),
            "elo": trial.suggest_float("w_elo", 0.00, 0.30),
        }
        # 正規化
        total = sum(raw_weights.values())
        weights = {k: v / total for k, v in raw_weights.items()}

        # ハイパーパラメータ探索
        params = {
            "sigmoid_k": trial.suggest_float("sigmoid_k", 1.0, 8.0),
            "sigmoid_m": trial.suggest_float("sigmoid_m", 0.15, 0.45),
            "base_home": trial.suggest_float("base_home", 0.30, 0.50),
            "base_draw": trial.suggest_float("base_draw", 0.15, 0.40),
            "form_n": trial.suggest_int("form_n", 3, 8),
            "draw_closeness_boost": trial.suggest_float("draw_closeness_boost", 0.0, 0.20),
            "elo_k": trial.suggest_float("elo_k", 10.0, 60.0),
            "elo_home_bonus": trial.suggest_float("elo_home_bonus", 20.0, 100.0),
        }
        params["base_away"] = 1.0 - params["base_home"] - params["base_draw"]
        if params["base_away"] < 0.10:
            return 999.0  # 不正な構成

        result = run_backtest(results, weights, params, val_ratio=0.3)
        val_metrics = result.get("val_metrics", {})
        val_acc = val_metrics.get("accuracy", 0.0)
        val_ll = val_metrics.get("log_loss", 999.0)

        # 停止条件チェック (log_loss基準: 小さいほど良い)
        if best_val_acc < 0 or val_ll < best_val_acc:
            improvement = best_val_acc - val_ll if best_val_acc > 0 else 1.0
            best_val_acc = val_ll  # actually stores best_val_ll
            no_improve_count = 0
        else:
            improvement = 0.0
            no_improve_count += 1

        recent_improvements.append(improvement)
        if len(recent_improvements) > 10:
            recent_improvements.pop(0)

        return val_ll

    # Optuna study
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # コールバックで停止条件を確認
    class StopCallback:
        def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
            nonlocal no_improve_count, recent_improvements
            if no_improve_count >= patience:
                logger.info("Early stopping: %d trials without improvement", patience)
                study.stop()
            if len(recent_improvements) >= 10 and all(imp < min_improvement for imp in recent_improvements):
                logger.info("Early stopping: improvement below threshold")
                study.stop()

    study.optimize(objective, n_trials=n_trials, callbacks=[StopCallback()])

    # 最良パラメータで再評価
    best = study.best_trial
    raw_w = {
        "team_strength": best.params["w_team_strength"],
        "attack_rate": best.params["w_attack_rate"],
        "defense_rate": best.params["w_defense_rate"],
        "recent_form": best.params["w_recent_form"],
        "home_advantage": best.params["w_home_advantage"],
        "capital_power": best.params["w_capital_power"],
        "elo": best.params["w_elo"],
    }
    total = sum(raw_w.values())
    best_weights = {k: round(v / total, 4) for k, v in raw_w.items()}

    best_params = {
        "sigmoid_k": best.params["sigmoid_k"],
        "sigmoid_m": best.params["sigmoid_m"],
        "base_home": best.params["base_home"],
        "base_draw": best.params["base_draw"],
        "base_away": round(1.0 - best.params["base_home"] - best.params["base_draw"], 4),
        "form_n": best.params["form_n"],
        "draw_closeness_boost": best.params["draw_closeness_boost"],
        "elo_k": best.params["elo_k"],
        "elo_home_bonus": best.params["elo_home_bonus"],
    }

    return {
        "best_weights": best_weights,
        "best_params": best_params,
        "best_val_log_loss": best.value,
        "n_trials": len(study.trials),
        "study": study,
    }


# ────────────────────────────────────────────────────────
# 7. メインエントリポイント
# ────────────────────────────────────────────────────────

# 現行v5ベースライン重み (有効パラメータのみ抽出・正規化)
BASELINE_WEIGHTS = {
    "team_strength": 0.2044,
    "attack_rate": 0.1363,
    "defense_rate": 0.1022,
    "recent_form": 0.2559,
    "home_advantage": 0.1363,
    "capital_power": 0.1704,
}
# 元の重み合計: 0.1224+0.0816+0.0612+0.1531+0.0816+0.1020 = 0.6019
# 正規化: 各/0.6019 → 上記

BASELINE_PARAMS = {
    "sigmoid_k": 3.0,
    "sigmoid_m": 0.30,
    "base_home": 0.40,
    "base_draw": 0.25,
    "base_away": 0.35,
    "form_n": 5,
}


def main():
    parser = argparse.ArgumentParser(description="Jリーグ予測バックテスト")
    parser.add_argument("--division", default="j1", help="対象リーグ")
    parser.add_argument("--optimize", action="store_true", help="Optuna最適化実行")
    parser.add_argument("--n-trials", type=int, default=200, help="最大試行回数")
    parser.add_argument("--patience", type=int, default=20, help="改善なし停止回数")
    parser.add_argument("--experiment-id", default=None, help="実験ID")
    parser.add_argument("--notes", default="", help="実験メモ")
    args = parser.parse_args()

    logger.info("=== Jリーグ予測バックテスト ===")
    logger.info("Division: %s", args.division)

    # データ取得
    logger.info("過去試合データ取得中...")
    results = fetch_past_results(args.division)
    logger.info("取得試合数: %d", len(results))

    if len(results) < 10:
        logger.error("試合データが不足しています (最低10試合必要)")
        return

    # ベースライン評価
    logger.info("\n--- ベースライン評価 ---")
    baseline = run_backtest(results, BASELINE_WEIGHTS, BASELINE_PARAMS)

    exp_id = args.experiment_id or f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("\n========== ベースライン結果 ==========")
    print(f"評価可能試合数: {baseline['n_total']} (train={baseline['n_train']}, val={baseline['n_val']})")
    for split_name in ["train_metrics", "val_metrics", "all_metrics"]:
        m = baseline.get(split_name, {})
        if m and "accuracy" in m:
            print(f"\n  [{split_name}]")
            print(f"    accuracy:    {m['accuracy']:.4f}")
            print(f"    macro F1:    {m['f1_macro']:.4f}")
            print(f"    log loss:    {m['log_loss']:.4f}")
            print(f"    Brier score: {m['brier_score']:.4f}")
            print(f"    分布: {m['class_distribution']}")
            if "confusion_matrix" in m:
                print(f"    混同行列 [home/draw/away]:")
                for row in m["confusion_matrix"]:
                    print(f"      {row}")

    # ログ保存
    save_experiment_log(
        experiment_id=exp_id,
        phase="baseline",
        weights=BASELINE_WEIGHTS,
        features=list(BASELINE_WEIGHTS.keys()),
        val_metrics=baseline.get("val_metrics", {}),
        train_metrics=baseline.get("train_metrics", {}),
        n_train=baseline["n_train"],
        n_val=baseline["n_val"],
        notes=args.notes or "v5 baseline (dead params removed, normalized)",
    )

    # 詳細結果保存
    detail_path = RESULTS_DIR / f"{exp_id}_detail.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        # numpyの型をシリアライズ可能にする
        serializable = {
            k: v for k, v in baseline.items()
            if k != "predictions"
        }
        serializable["predictions_sample"] = baseline["predictions"][:10]
        json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)
    logger.info("詳細結果保存: %s", detail_path)

    if not args.optimize:
        return

    # Optuna最適化
    logger.info("\n--- Optuna最適化開始 ---")
    logger.info("最大試行回数: %d, patience: %d", args.n_trials, args.patience)

    opt_result = optimize_weights(
        results,
        n_trials=args.n_trials,
        patience=args.patience,
    )

    print(f"\n========== 最適化結果 ==========")
    print(f"試行回数: {opt_result['n_trials']}")
    print(f"最良 val log_loss: {opt_result['best_val_log_loss']:.4f}")
    print(f"\n最良重み:")
    for k, v in opt_result["best_weights"].items():
        print(f"  {k}: {v:.4f}")
    print(f"\n最良パラメータ:")
    for k, v in opt_result["best_params"].items():
        print(f"  {k}: {v}")

    # 最良パラメータで再評価
    best_result = run_backtest(
        results,
        opt_result["best_weights"],
        opt_result["best_params"],
    )

    print(f"\n--- 最良モデル評価 ---")
    for split_name in ["train_metrics", "val_metrics"]:
        m = best_result.get(split_name, {})
        if m and "accuracy" in m:
            print(f"\n  [{split_name}]")
            print(f"    accuracy:    {m['accuracy']:.4f}")
            print(f"    macro F1:    {m['f1_macro']:.4f}")
            print(f"    log loss:    {m['log_loss']:.4f}")
            print(f"    Brier score: {m['brier_score']:.4f}")

    # 改善比較
    b_val = baseline.get("val_metrics", {})
    o_val = best_result.get("val_metrics", {})
    if b_val and o_val and "accuracy" in b_val and "accuracy" in o_val:
        print(f"\n--- 改善サマリ ---")
        print(f"  accuracy:  {b_val['accuracy']:.4f} → {o_val['accuracy']:.4f} ({o_val['accuracy']-b_val['accuracy']:+.4f})")
        print(f"  macro F1:  {b_val['f1_macro']:.4f} → {o_val['f1_macro']:.4f} ({o_val['f1_macro']-b_val['f1_macro']:+.4f})")
        print(f"  log loss:  {b_val['log_loss']:.4f} → {o_val['log_loss']:.4f} ({o_val['log_loss']-b_val['log_loss']:+.4f})")

    # 最適化ログ保存
    opt_exp_id = f"optuna_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_experiment_log(
        experiment_id=opt_exp_id,
        phase="phase2_optuna",
        weights=opt_result["best_weights"],
        features=list(opt_result["best_weights"].keys()),
        val_metrics=best_result.get("val_metrics", {}),
        train_metrics=best_result.get("train_metrics", {}),
        n_train=best_result["n_train"],
        n_val=best_result["n_val"],
        notes=f"Optuna {opt_result['n_trials']} trials, best_ll={opt_result['best_val_log_loss']:.4f}",
    )

    # 最適化詳細保存
    opt_detail_path = RESULTS_DIR / f"{opt_exp_id}_detail.json"
    with open(opt_detail_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_weights": opt_result["best_weights"],
            "best_params": opt_result["best_params"],
            "best_val_log_loss": opt_result["best_val_log_loss"],
            "n_trials": opt_result["n_trials"],
            "baseline_val_metrics": b_val,
            "optimized_val_metrics": o_val,
        }, f, ensure_ascii=False, indent=2, default=str)

    logger.info("最適化詳細保存: %s", opt_detail_path)


if __name__ == "__main__":
    main()
