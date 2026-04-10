"""
scripts/skellam_model.py - Dixon-Coles xG-Skellam モデル

ホーム/アウェイの期待得点をポアソン分布でモデル化し、
Skellam分布 (2つの独立ポアソンの差) の考え方で3クラス確率を算出する。

P(home_goals = i) = Poisson(i | λ_home)
P(away_goals = j) = Poisson(j | λ_away)
P(home win) = Σ P(i) * P(j) for i > j
P(draw)    = Σ P(i) * P(j) for i = j
P(away win) = Σ P(i) * P(j) for i < j

参考: Dixon & Coles (1997), Modelling Association Football Scores
"""

from __future__ import annotations

import math
from typing import Any


# J1 リーグ平均得点/試合 (2024-2025実績ベース)
LEAGUE_AVG_GOALS = 1.30
MAX_GOALS = 8  # 列挙する最大得点 (0-8)

# Dixon-Colesの低スコア補正 (draw の確率が過小評価される問題を緩和)
RHO = -0.08  # 負の値で draw 0-0, 1-1 の確率をブースト


def _poisson_pmf(k: int, lam: float) -> float:
    """ポアソン分布の確率質量関数"""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _dixon_coles_correction(i: int, j: int, lam: float, mu: float, rho: float) -> float:
    """
    Dixon-Colesの低スコア補正項 τ(i, j, λ, μ, ρ)
    0-0, 1-0, 0-1, 1-1 の4ケースのみ補正
    """
    if i == 0 and j == 0:
        return 1.0 - (lam * mu * rho)
    if i == 0 and j == 1:
        return 1.0 + (lam * rho)
    if i == 1 and j == 0:
        return 1.0 + (mu * rho)
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def predict_skellam(
    home_stats: dict,
    away_stats: dict,
    home_advantage: float = 0.25,
    league_avg: float = LEAGUE_AVG_GOALS,
    use_dc_correction: bool = True,
) -> dict[str, Any]:
    """
    Skellamモデルによる予測。

    Parameters
    ----------
    home_stats, away_stats : チーム統計 (得点/試合, 失点/試合を含む)
    home_advantage : ホームアドバンテージ係数 (λ に乗算)
    league_avg : リーグ平均得点/試合
    use_dc_correction : Dixon-Coles低スコア補正を適用するか

    Returns
    -------
    {
        "home_win_prob": int (0-100),
        "draw_prob": int,
        "away_win_prob": int,
        "lambda_home": float (期待得点),
        "lambda_away": float,
        "model_version": "xg_skellam",
    }
    """
    # 得点率と失点率を抽出
    def _rate(s: dict, key: str) -> float:
        games = max(float(s.get("試合", 1)), 1)
        val = float(s.get(key, league_avg))
        return val / games

    h_gf = _rate(home_stats, "得点")
    h_ga = _rate(home_stats, "失点")
    a_gf = _rate(away_stats, "得点")
    a_ga = _rate(away_stats, "失点")

    # シーズン序盤シュリンク (試合数が少ないときはリーグ平均へ回帰)
    h_games = max(float(home_stats.get("試合", 1)), 1)
    a_games = max(float(away_stats.get("試合", 1)), 1)
    h_shrink = max(0.0, 1.0 - h_games / 10.0)
    a_shrink = max(0.0, 1.0 - a_games / 10.0)
    h_gf = h_gf * (1 - h_shrink) + league_avg * h_shrink
    h_ga = h_ga * (1 - h_shrink) + league_avg * h_shrink
    a_gf = a_gf * (1 - a_shrink) + league_avg * a_shrink
    a_ga = a_ga * (1 - a_shrink) + league_avg * a_shrink

    # 攻撃力 α_i = GF_i / league_avg
    # 守備弱さ β_i = GA_i / league_avg
    alpha_h = h_gf / league_avg
    beta_h = h_ga / league_avg
    alpha_a = a_gf / league_avg
    beta_a = a_ga / league_avg

    # 期待得点
    # λ_home = α_home * β_away * league_avg * (1 + home_advantage)
    # μ_away = α_away * β_home * league_avg
    lam_home = alpha_h * beta_a * league_avg * (1.0 + home_advantage)
    lam_away = alpha_a * beta_h * league_avg

    # クリップ (極端な値を抑制)
    lam_home = max(0.1, min(5.0, lam_home))
    lam_away = max(0.1, min(5.0, lam_away))

    # 確率テーブル計算 (0〜MAX_GOALS)
    p_home_win = 0.0
    p_draw = 0.0
    p_away_win = 0.0

    for i in range(MAX_GOALS + 1):
        p_i = _poisson_pmf(i, lam_home)
        for j in range(MAX_GOALS + 1):
            p_j = _poisson_pmf(j, lam_away)
            p_ij = p_i * p_j
            if use_dc_correction:
                p_ij *= _dixon_coles_correction(i, j, lam_home, lam_away, RHO)
            if i > j:
                p_home_win += p_ij
            elif i == j:
                p_draw += p_ij
            else:
                p_away_win += p_ij

    # 正規化
    total = p_home_win + p_draw + p_away_win
    if total <= 0:
        return {
            "home_win_prob": 34, "draw_prob": 33, "away_win_prob": 33,
            "lambda_home": lam_home, "lambda_away": lam_away,
            "model_version": "xg_skellam",
        }
    p_home_win /= total
    p_draw /= total
    p_away_win /= total

    h_pct = round(p_home_win * 100)
    a_pct = round(p_away_win * 100)
    d_pct = 100 - h_pct - a_pct

    # 予想スコア (最頻モード)
    best_score = (1, 1)
    best_p = 0
    for i in range(4):
        for j in range(4):
            p_ij = _poisson_pmf(i, lam_home) * _poisson_pmf(j, lam_away)
            if p_ij > best_p:
                best_p = p_ij
                best_score = (i, j)

    return {
        "home_win_prob": h_pct,
        "draw_prob": d_pct,
        "away_win_prob": a_pct,
        "predicted_score": f"{best_score[0]}-{best_score[1]}",
        "lambda_home": round(lam_home, 3),
        "lambda_away": round(lam_away, 3),
        "model_version": "xg_skellam",
    }
