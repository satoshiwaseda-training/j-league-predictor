"""
scripts/backtest_runner.py - Jリーグ予測モデル バックテスト基盤 v2

3層評価設計:
  A. Walk-forward: 2024(train) → 2025(validation) を節単位で逐次評価
  B. Strict holdout: 2026 第11節以降 (最適化に一切使わない)
  C. 2026逐次評価: 第1-10節のadaptation window

未来情報リークを防ぐため、各試合の予測には「その試合前」のデータのみを使用。

Usage:
    python scripts/backtest_runner.py                        # 3層ベースライン評価
    python scripts/backtest_runner.py --optimize              # Optuna最適化 (train→val)
    python scripts/backtest_runner.py --baselines             # 全ベースライン比較
    python scripts/backtest_runner.py --experiment-id exp1 --notes "description"
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).parent.parent
LOG_PATH = ROOT / "experiment_logs.csv"
RESULTS_DIR = ROOT / "backtest_results"
RESULTS_DIR.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════
# 1. チーム状態の再構築 (時系列厳守)
# ════════════════════════════════════════════════════════

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
    recent_results: list = field(default_factory=list)

    @property
    def goal_diff(self) -> int:
        return self.goals_for - self.goals_against

    def to_stats_dict(self, rank: int = 0) -> dict:
        """predict_logic.py の score_* 関数が期待する dict 形式"""
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
        return self.recent_results[-n:] if self.recent_results else []


def rebuild_states(results: list[dict], up_to_index: int) -> dict[str, TeamState]:
    """results[0:up_to_index] のみで各チームの累積状態を構築"""
    states: dict[str, TeamState] = defaultdict(TeamState)
    for i in range(up_to_index):
        r = results[i]
        h, a = r["home"], r["away"]
        hs, as_ = int(r["home_score"]), int(r["away_score"])
        for team, gf, ga in [(h, hs, as_), (a, as_, hs)]:
            s = states[team]
            s.games += 1
            s.goals_for += gf
            s.goals_against += ga
            if gf > ga:
                s.wins += 1; s.points += 3; s.recent_results.append("W")
            elif gf == ga:
                s.draws += 1; s.points += 1; s.recent_results.append("D")
            else:
                s.losses += 1; s.recent_results.append("L")
    return states


def compute_ranks(states: dict[str, TeamState]) -> dict[str, int]:
    ranked = sorted(states, key=lambda t: (-states[t].points, -states[t].goal_diff, -states[t].goals_for))
    return {team: i + 1 for i, team in enumerate(ranked)}


# ════════════════════════════════════════════════════════
# 2. ELO レーティング
# ════════════════════════════════════════════════════════

class EloSystem:
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
        rh = self.get(home) + self.home_bonus
        ra = self.get(away)
        eh = self.expected(rh, ra)
        sh = 1.0 if winner == "home" else (0.5 if winner == "draw" else 0.0)
        self.ratings[home] = self.get(home) + self.k * (sh - eh)
        self.ratings[away] = self.get(away) + self.k * ((1.0 - sh) - (1.0 - eh))

    def score_pair(self, home: str, away: str) -> tuple[float, float]:
        rh = self.get(home) + self.home_bonus
        ra = self.get(away)
        eh = self.expected(rh, ra)
        return eh, 1.0 - eh

    def clone(self) -> "EloSystem":
        e = EloSystem(self.k, self.initial, self.home_bonus)
        e.ratings = dict(self.ratings)
        return e


def build_elo(results: list[dict], up_to_index: int,
              k: float = 32.0, home_bonus: float = 50.0) -> EloSystem:
    elo = EloSystem(k=k, home_bonus=home_bonus)
    for i in range(up_to_index):
        r = results[i]
        elo.update(r["home"], r["away"], r["winner"])
    return elo


# ════════════════════════════════════════════════════════
# 3. 予測エンジン群 (全て再現可能・Gemini不使用)
# ════════════════════════════════════════════════════════

def predict_current_model(
    home: str, away: str,
    h_stats: dict, a_stats: dict,
    h_form: list[str], a_form: list[str],
    weights: dict[str, float],
    params: dict[str, Any],
    elo: EloSystem | None = None,
) -> dict[str, float]:
    """現行ロジック (重み付き線形モデル + シグモイド変換)"""
    from predict_logic import (
        score_team_strength, score_attack_rate, score_defense_rate,
        score_recent_form, score_home_advantage, score_capital_power,
    )
    p = params
    form_n = p.get("form_n", 5)

    h_str, a_str = score_team_strength(h_stats, a_stats)
    h_atk, a_atk = score_attack_rate(h_stats, a_stats)
    h_def, a_def = score_defense_rate(h_stats, a_stats)
    h_f = score_recent_form(h_form[-form_n:])
    a_f = score_recent_form(a_form[-form_n:])
    h_ha = score_home_advantage()
    a_ha = 1.0 - h_ha
    h_cap, a_cap = score_capital_power(home, away)

    scores: dict[str, tuple[float, float]] = {
        "team_strength": (h_str, a_str),
        "attack_rate":   (h_atk, a_atk),
        "defense_rate":  (h_def, a_def),
        "recent_form":   (h_f, a_f),
        "home_advantage": (h_ha, a_ha),
        "capital_power": (h_cap, a_cap),
    }
    if elo and weights.get("elo", 0) > 0:
        scores["elo"] = elo.score_pair(home, away)

    raw_adv = 0.0
    tw = 0.0
    for name, (hs, as_) in scores.items():
        w = weights.get(name, 0.0)
        raw_adv += (hs - as_) * w
        tw += w
    if tw > 0 and abs(tw - 1.0) > 0.01:
        raw_adv /= tw

    sk = p.get("sigmoid_k", 3.0)
    sm = p.get("sigmoid_m", 0.30)
    bh = p.get("base_home", 0.40)
    bd = p.get("base_draw", 0.25)
    ba = p.get("base_away", 0.35)
    dcb = p.get("draw_closeness_boost", 0.0)

    shift = math.tanh(raw_adv * sk) * sm
    h = max(0.05, min(0.90, bh + shift))
    a = max(0.05, min(0.90, ba - shift))
    closeness = max(0.0, 1.0 - abs(raw_adv) * 2)
    d = max(0.05, min(0.50, (1.0 - h - a) + dcb * closeness))
    t = h + d + a
    return {"home": h / t, "draw": d / t, "away": a / t}


def predict_always_home(**_kw) -> dict[str, float]:
    """ベースライン: 常にホーム勝ち"""
    return {"home": 0.60, "draw": 0.20, "away": 0.20}


def predict_elo_only(elo: EloSystem, home: str, away: str, **_kw) -> dict[str, float]:
    """ベースライン: 単純ELO + 接近時draw"""
    eh, ea = elo.score_pair(home, away)
    closeness = 1.0 - abs(eh - ea) * 2  # 接近度
    d = max(0.10, 0.20 + max(0.0, closeness) * 0.15)
    h = eh * (1.0 - d)
    a = ea * (1.0 - d)
    t = h + d + a
    return {"home": h / t, "draw": d / t, "away": a / t}


def predict_form_only(h_form: list[str], a_form: list[str], **_kw) -> dict[str, float]:
    """ベースライン: 直近フォームのみ"""
    from predict_logic import score_recent_form
    hf = score_recent_form(h_form[-5:])
    af = score_recent_form(a_form[-5:])
    diff = hf - af + 0.05  # home advantage offset
    shift = math.tanh(diff * 3.0) * 0.25
    h = max(0.10, 0.40 + shift)
    a = max(0.10, 0.35 - shift)
    d = max(0.10, 1.0 - h - a)
    t = h + d + a
    return {"home": h / t, "draw": d / t, "away": a / t}


def predict_uniform(**_kw) -> dict[str, float]:
    """ベースライン: 均等確率 (argmax→draw tie-break回避のためhome微優勢)"""
    return {"home": 0.334, "draw": 0.333, "away": 0.333}


def predict_prior(**_kw) -> dict[str, float]:
    """ベースライン: J1全体の事前確率 (2024-2025平均)"""
    return {"home": 0.413, "draw": 0.265, "away": 0.322}


def predict_draw_aware(**_kw) -> dict[str, float]:
    """ベースライン: draw重視の固定確率 (draw率~26%を反映)"""
    return {"home": 0.38, "draw": 0.27, "away": 0.35}


# ════════════════════════════════════════════════════════
# 4. 評価メトリクス
# ════════════════════════════════════════════════════════

LABELS = ["away", "draw", "home"]  # sklearn lexicographic order


def compute_metrics(y_true: list[str], y_pred: list[str],
                    y_prob: list[dict[str, float]]) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
    n = len(y_true)
    if n == 0:
        return {"n_samples": 0}

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    pm = np.array([[p.get(c, 1e-3) for c in LABELS] for p in y_prob])
    pm = np.clip(pm, 0.01, 0.99)
    pm = pm / pm.sum(axis=1, keepdims=True)
    ll = log_loss(y_true, pm, labels=LABELS)
    y_oh = np.array([[1 if y == c else 0 for c in LABELS] for y in y_true])
    brier = float(np.mean(np.sum((pm - y_oh) ** 2, axis=1)))
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    # クラス別precision/recall
    class_metrics = {}
    for i, c in enumerate(LABELS):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(3)) - tp
        fn = sum(cm[i]) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        class_metrics[c] = {"precision": round(prec, 4), "recall": round(rec, 4),
                            "support": int(sum(cm[i]))}

    return {
        "n_samples": n,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
        "log_loss": round(ll, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": cm.tolist(),
        "class_metrics": class_metrics,
        "class_distribution": {c: sum(1 for y in y_true if y == c) for c in LABELS},
        "predicted_distribution": {c: sum(1 for y in y_pred if y == c) for c in LABELS},
    }


# ════════════════════════════════════════════════════════
# 5. Walk-forward バックテスト
# ════════════════════════════════════════════════════════

def run_walk_forward(
    all_results: list[dict],
    eval_season: int,
    predictor: str = "current",
    weights: dict[str, float] | None = None,
    params: dict[str, Any] | None = None,
    min_games: int = 1,
) -> dict[str, Any]:
    """
    all_results 内で season == eval_season の試合を逐次予測する。
    各試合の予測には「その試合以前」の全データ (他シーズン含む) を使用。

    Parameters
    ----------
    all_results : 日付順ソート済み全試合 (train + eval 含む)
    eval_season : 評価対象シーズン (例: 2025)
    predictor : "current" / "always_home" / "elo_only" / "form_only" / "uniform" / "prior"
    weights : 現行モデル用重み
    params : 現行モデル用パラメータ
    min_games : 最低試合数

    Returns
    -------
    {"metrics": {...}, "predictions": [...]}
    """
    weights = weights or BASELINE_WEIGHTS
    params = params or BASELINE_PARAMS
    predictions = []

    for idx, match in enumerate(all_results):
        if match.get("season") != eval_season:
            continue
        actual = match.get("winner")
        if not actual:
            continue

        home, away = match["home"], match["away"]
        states = rebuild_states(all_results, idx)
        ranks = compute_ranks(states)

        hs = states.get(home)
        as_ = states.get(away)
        if not hs or hs.games < min_games or not as_ or as_.games < min_games:
            continue

        h_stats = hs.to_stats_dict(ranks.get(home, 99))
        a_stats = as_.to_stats_dict(ranks.get(away, 99))
        h_form = hs.get_form(params.get("form_n", 5))
        a_form = as_.get_form(params.get("form_n", 5))

        # ELO構築
        elo_k = params.get("elo_k", 32.0)
        elo_hb = params.get("elo_home_bonus", 50.0)
        elo = build_elo(all_results, idx, k=elo_k, home_bonus=elo_hb)

        # 予測
        if predictor == "current":
            probs = predict_current_model(home, away, h_stats, a_stats, h_form, a_form,
                                          weights, params, elo)
        elif predictor == "always_home":
            probs = predict_always_home()
        elif predictor == "elo_only":
            probs = predict_elo_only(elo, home, away)
        elif predictor == "form_only":
            probs = predict_form_only(h_form, a_form)
        elif predictor == "uniform":
            probs = predict_uniform()
        elif predictor == "prior":
            probs = predict_prior()
        elif predictor == "draw_aware":
            probs = predict_draw_aware()
        else:
            probs = predict_current_model(home, away, h_stats, a_stats, h_form, a_form,
                                          weights, params, elo)

        pred = max(probs, key=probs.get)
        predictions.append({
            "idx": idx, "date": match.get("date", ""), "section": match.get("section", 0),
            "home": home, "away": away,
            "actual": actual, "predicted": pred,
            "prob_home": round(probs["home"], 4),
            "prob_draw": round(probs["draw"], 4),
            "prob_away": round(probs["away"], 4),
        })

    if not predictions:
        return {"metrics": {"n_samples": 0}, "predictions": []}

    yt = [p["actual"] for p in predictions]
    yp = [p["predicted"] for p in predictions]
    ypr = [{"home": p["prob_home"], "draw": p["prob_draw"], "away": p["prob_away"]} for p in predictions]

    return {"metrics": compute_metrics(yt, yp, ypr), "predictions": predictions}


# ════════════════════════════════════════════════════════
# 6. 実験ログ
# ════════════════════════════════════════════════════════

LOG_FIELDS = [
    "timestamp", "experiment_id", "git_hash", "predictor", "eval_season",
    "parameter_set", "features_used",
    "accuracy", "f1_macro", "log_loss", "brier_score",
    "n_samples", "draw_predicted", "notes",
]


def _git_hash() -> str:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(ROOT), text=True).strip()
    except Exception:
        return "unknown"


def save_log(experiment_id: str, predictor: str, eval_season: int,
             metrics: dict, weights: dict | None = None,
             features: list[str] | None = None, notes: str = "") -> None:
    write_header = not LOG_PATH.exists()
    row = {
        "timestamp": datetime.now().isoformat(),
        "experiment_id": experiment_id,
        "git_hash": _git_hash(),
        "predictor": predictor,
        "eval_season": eval_season,
        "parameter_set": json.dumps(weights or {}, ensure_ascii=False),
        "features_used": json.dumps(features or [], ensure_ascii=False),
        "accuracy": metrics.get("accuracy", ""),
        "f1_macro": metrics.get("f1_macro", ""),
        "log_loss": metrics.get("log_loss", ""),
        "brier_score": metrics.get("brier_score", ""),
        "n_samples": metrics.get("n_samples", ""),
        "draw_predicted": metrics.get("predicted_distribution", {}).get("draw", 0),
        "notes": notes,
    }
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


# ════════════════════════════════════════════════════════
# 7. Optuna 最適化 (train=2024 → val=2025)
# ════════════════════════════════════════════════════════

def optimize(all_results: list[dict], n_trials: int = 200, patience: int = 40) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    best_ll = float("inf")
    no_improve = 0

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_ll, no_improve
        rw = {
            "team_strength":  trial.suggest_float("w_ts", 0.01, 0.40),
            "attack_rate":    trial.suggest_float("w_atk", 0.01, 0.25),
            "defense_rate":   trial.suggest_float("w_def", 0.01, 0.25),
            "recent_form":    trial.suggest_float("w_rf", 0.01, 0.40),
            "home_advantage": trial.suggest_float("w_ha", 0.01, 0.25),
            "capital_power":  trial.suggest_float("w_cp", 0.00, 0.25),
            "elo":            trial.suggest_float("w_elo", 0.00, 0.30),
        }
        t = sum(rw.values())
        weights = {k: v / t for k, v in rw.items()}

        params = {
            "sigmoid_k": trial.suggest_float("sk", 1.0, 8.0),
            "sigmoid_m": trial.suggest_float("sm", 0.15, 0.45),
            "base_home": trial.suggest_float("bh", 0.30, 0.50),
            "base_draw": trial.suggest_float("bd", 0.15, 0.35),
            "form_n": trial.suggest_int("fn", 3, 8),
            "draw_closeness_boost": trial.suggest_float("dcb", 0.0, 0.20),
            "elo_k": trial.suggest_float("ek", 10.0, 60.0),
            "elo_home_bonus": trial.suggest_float("ehb", 20.0, 100.0),
        }
        params["base_away"] = 1.0 - params["base_home"] - params["base_draw"]
        if params["base_away"] < 0.10:
            return 999.0

        res = run_walk_forward(all_results, eval_season=2025,
                               predictor="current", weights=weights, params=params)
        ll = res["metrics"].get("log_loss", 999.0)

        if ll < best_ll:
            best_ll = ll
            no_improve = 0
        else:
            no_improve += 1
        return ll

    class StopCB:
        def __call__(self, study, trial):
            if no_improve >= patience:
                logger.info("Early stopping at trial %d (patience=%d)", trial.number, patience)
                study.stop()

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, callbacks=[StopCB()])

    bp = study.best_trial.params
    rw = {
        "team_strength": bp["w_ts"], "attack_rate": bp["w_atk"],
        "defense_rate": bp["w_def"], "recent_form": bp["w_rf"],
        "home_advantage": bp["w_ha"], "capital_power": bp["w_cp"],
        "elo": bp["w_elo"],
    }
    t = sum(rw.values())
    best_weights = {k: round(v / t, 4) for k, v in rw.items()}
    best_params = {
        "sigmoid_k": bp["sk"], "sigmoid_m": bp["sm"],
        "base_home": bp["bh"], "base_draw": bp["bd"],
        "base_away": round(1.0 - bp["bh"] - bp["bd"], 4),
        "form_n": bp["fn"],
        "draw_closeness_boost": bp["dcb"],
        "elo_k": bp["ek"], "elo_home_bonus": bp["ehb"],
    }
    return {
        "best_weights": best_weights,
        "best_params": best_params,
        "best_log_loss": study.best_value,
        "n_trials": len(study.trials),
    }


# ════════════════════════════════════════════════════════
# 8. 定数
# ════════════════════════════════════════════════════════

# 現行 v5 重み (有効6パラメータ正規化済み)
BASELINE_WEIGHTS: dict[str, float] = {
    "team_strength": 0.2044,
    "attack_rate":   0.1363,
    "defense_rate":  0.1022,
    "recent_form":   0.2559,
    "home_advantage": 0.1363,
    "capital_power": 0.1704,
}

BASELINE_PARAMS: dict[str, Any] = {
    "sigmoid_k": 3.0,
    "sigmoid_m": 0.30,
    "base_home": 0.40,
    "base_draw": 0.25,
    "base_away": 0.35,
    "form_n": 5,
    "draw_closeness_boost": 0.0,
    "elo_k": 32.0,
    "elo_home_bonus": 50.0,
}

ALL_PREDICTORS = ["current", "always_home", "elo_only", "form_only", "uniform", "prior", "draw_aware"]


# ════════════════════════════════════════════════════════
# 9. メイン
# ════════════════════════════════════════════════════════

def _print_metrics(label: str, m: dict) -> None:
    if m.get("n_samples", 0) == 0:
        print(f"  [{label}] No data")
        return
    print(f"  [{label}] n={m['n_samples']}")
    print(f"    accuracy:    {m['accuracy']:.4f}")
    print(f"    macro F1:    {m['f1_macro']:.4f}")
    print(f"    log loss:    {m['log_loss']:.4f}")
    print(f"    Brier:       {m['brier_score']:.4f}")
    print(f"    actual dist: {m['class_distribution']}")
    print(f"    pred dist:   {m['predicted_distribution']}")
    cm = m.get("confusion_matrix", [])
    if cm:
        print(f"    confusion [away/draw/home]:")
        for row in cm:
            print(f"      {row}")


def main():
    parser = argparse.ArgumentParser(description="Jリーグ予測バックテスト v2")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    # データ取得
    logger.info("Fetching multi-season data...")
    from data_fetcher import get_multi_season_results
    all_results = get_multi_season_results([2024, 2025], "j1")
    logger.info("Total matches: %d", len(all_results))

    # シーズン別集計
    by_season = Counter(r.get("season") for r in all_results)
    for s, c in sorted(by_season.items()):
        logger.info("  Season %s: %d matches", s, c)

    # ─── ベースライン全比較 ───
    if args.baselines:
        print("\n" + "=" * 60)
        print("  BASELINE COMPARISON (eval=2025, train<=2024)")
        print("=" * 60)

        results_table = []
        for pred_name in ALL_PREDICTORS:
            res = run_walk_forward(all_results, eval_season=2025, predictor=pred_name)
            m = res["metrics"]
            results_table.append((pred_name, m))
            _print_metrics(pred_name, m)
            save_log(f"baseline_{pred_name}", pred_name, 2025, m,
                     notes=f"Baseline comparison: {pred_name}")
            print()

        # サマリ表
        print("\n--- Summary ---")
        print(f"{'Predictor':<15} {'Acc':>7} {'F1':>7} {'LogL':>7} {'Brier':>7} {'Draw#':>6}")
        print("-" * 55)
        for name, m in results_table:
            if m.get("n_samples", 0) == 0:
                continue
            dp = m.get("predicted_distribution", {}).get("draw", 0)
            print(f"{name:<15} {m['accuracy']:>7.4f} {m['f1_macro']:>7.4f} "
                  f"{m['log_loss']:>7.4f} {m['brier_score']:>7.4f} {dp:>6}")
        return

    # ─── 3層評価 ───
    print("\n" + "=" * 60)
    print("  3-LAYER EVALUATION")
    print("=" * 60)

    # Layer A: Walk-forward 2025 (trained on ≤2024)
    print("\n--- Layer A: Walk-forward 2025 (train<=2024) ---")
    res_2025 = run_walk_forward(all_results, eval_season=2025, predictor="current")
    _print_metrics("val_2025", res_2025["metrics"])

    exp_id = args.experiment_id or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_log(exp_id + "_val2025", "current", 2025, res_2025["metrics"],
             weights=BASELINE_WEIGHTS, features=list(BASELINE_WEIGHTS.keys()),
             notes=args.notes or "3-layer eval: val 2025")

    # Layer B: 2026 (strict holdout if available)
    seasons_available = set(r.get("season") for r in all_results)
    if 2026 in seasons_available:
        print("\n--- Layer B: 2026 adaptation window (第1-10節) ---")
        res_2026 = run_walk_forward(all_results, eval_season=2026, predictor="current")
        _print_metrics("2026_adapt", res_2026["metrics"])
        save_log(exp_id + "_2026adapt", "current", 2026, res_2026["metrics"],
                 weights=BASELINE_WEIGHTS, features=list(BASELINE_WEIGHTS.keys()),
                 notes=args.notes or "3-layer eval: 2026 adaptation")
    else:
        print("\n--- Layer B: 2026 data not available ---")

    # Layer C: 2024 train-set fit check (overfit detection)
    print("\n--- Train-set fit check: 2024 ---")
    res_2024 = run_walk_forward(all_results, eval_season=2024, predictor="current")
    _print_metrics("train_2024", res_2024["metrics"])

    # 過学習チェック
    if res_2024["metrics"].get("accuracy") and res_2025["metrics"].get("accuracy"):
        gap = res_2024["metrics"]["accuracy"] - res_2025["metrics"]["accuracy"]
        print(f"\n  Overfit gap (train-val): {gap:+.4f}")
        if gap > 0.15:
            print("  WARNING: gap > 15pp → overfitting risk")

    # ─── Optuna最適化 ───
    if args.optimize:
        print("\n" + "=" * 60)
        print("  OPTUNA OPTIMIZATION (train=2024, val=2025)")
        print("=" * 60)
        logger.info("Starting optimization (max %d trials, patience %d)...",
                     args.n_trials, args.patience)

        opt = optimize(all_results, n_trials=args.n_trials, patience=args.patience)
        print(f"\nTrials: {opt['n_trials']}")
        print(f"Best val log_loss: {opt['best_log_loss']:.4f}")
        print(f"\nBest weights:")
        for k, v in opt["best_weights"].items():
            print(f"  {k}: {v:.4f}")
        print(f"\nBest params:")
        for k, v in opt["best_params"].items():
            print(f"  {k}: {v}")

        # 最良パラメータで再評価
        print("\n--- Optimized model on val=2025 ---")
        res_opt = run_walk_forward(all_results, eval_season=2025, predictor="current",
                                   weights=opt["best_weights"], params=opt["best_params"])
        _print_metrics("opt_val2025", res_opt["metrics"])

        # 2026 holdout
        if 2026 in seasons_available:
            print("\n--- Optimized model on 2026 holdout ---")
            res_opt26 = run_walk_forward(all_results, eval_season=2026, predictor="current",
                                         weights=opt["best_weights"], params=opt["best_params"])
            _print_metrics("opt_2026", res_opt26["metrics"])

        # 改善比較
        bv = res_2025["metrics"]
        ov = res_opt["metrics"]
        if bv.get("accuracy") and ov.get("accuracy"):
            print(f"\n--- Improvement (val 2025) ---")
            for k in ["accuracy", "f1_macro", "log_loss", "brier_score"]:
                d = ov.get(k, 0) - bv.get(k, 0)
                better = d > 0 if k in ("accuracy", "f1_macro") else d < 0
                mark = "+" if better else "-"
                print(f"  {k}: {bv[k]:.4f} → {ov[k]:.4f} ({d:+.4f} {mark})")

        opt_id = f"optuna_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_log(opt_id, "current_optimized", 2025, res_opt["metrics"],
                 weights=opt["best_weights"], features=list(opt["best_weights"].keys()),
                 notes=f"Optuna {opt['n_trials']} trials, ll={opt['best_log_loss']:.4f}")

        # 保存
        detail = RESULTS_DIR / f"{opt_id}_detail.json"
        with open(detail, "w", encoding="utf-8") as f:
            json.dump(opt, f, ensure_ascii=False, indent=2, default=str)
        logger.info("Saved: %s", detail)


if __name__ == "__main__":
    main()
