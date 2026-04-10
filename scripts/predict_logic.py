"""
scripts/predict_logic.py - Jリーグ予測コアロジック v3
Gemini 2.5 Flash 設計会議 (2026-03) + 論文ベース最適化

v3 追加パラメータ (Gemini設計):
  capital_power:   0.12  親会社収益・年俸総額スカッド質 (Peeters 2018: R²≈0.65-0.72)
  discipline_risk: 0.04  イエロー累積 → 守備断絶リスク
  attrition_rate:  0.04  損耗率 (怪我人比率) → 戦力維持能力
  match_interval:  0.04  試合間隔疲労 U字型 (Drust 2012, Pope 2015)
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from typing import Any

from dotenv import load_dotenv

# スクリプトから親ディレクトリをインポートできるようにする
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()
logger = logging.getLogger(__name__)

# ─── Primary Model: v7 refined (本番UI主力) ─────────────
# 重み合計 = 1.00
# 設計: 2026-04-09 (ELO重み再適合済み)
# 新ファクター3件 (チーム別ホームADV, H2H実データ, 昇格組補正) は維持
MODEL_WEIGHTS: dict[str, float] = {
    "team_strength":                0.1260,  # 勝点・順位差
    "attack_rate":                  0.0856,  # 得点率/試合 (Dixon-Coles lambda)
    "defense_rate":                 0.0640,  # 失点率/試合 (Dixon-Coles mu)
    "recent_form":                  0.1456,  # 直近フォームPPG
    "xg_for":                       0.0325,  # 期待得点 (攻撃力)
    "xg_against":                   0.0325,  # 期待失点 (守備力)
    "home_advantage":               0.0856,  # チーム別ホームADV
    "capital_power":                0.1063,  # 資本力 (R2=0.65-0.72 vs 勝点)
    "head_to_head":                 0.0325,  # H2H (実データ化済み)
    "discipline_risk":              0.0325,  # 規律・カード累積リスク
    "attrition_rate":               0.0325,  # 損耗率 (スカッド比率)
    "match_interval":               0.0325,  # 試合間隔疲労 U字型
    "injury_impact":                0.0098,  # 個別怪我絶対数
    "weather_fatigue":              0.0098,  # 天気疲労 (効果小)
    "travel_distance":              0.0000,  # 移動距離 (実質0)
    "set_piece_conversion":         0.0000,  # (inactive)
    "match_day_motivation":         0.0000,  # (inactive)
    "tactical_adaptability":        0.0000,  # (inactive)
    "expected_goals_difference":    0.0423,  # xGD
    "player_availability_impact":   0.0000,  # (inactive)
    "match_trend":                  0.0000,  # (inactive)
    "referee_tendency":             0.0000,  # (inactive)
    "elo":                          0.1300,  # ELOレーティング (v7値)
}

# ─── Shadow Model: v8.1 (内部ログ・比較用, UIには非表示) ──
# 重み合計 = 1.00
# val=2025で logL/Brier が過去最良 (1.045/0.628)
# UIには使わず、prediction_store に shadow として保存
V8_1_MODEL_WEIGHTS: dict[str, float] = {
    "team_strength":                0.1158,
    "attack_rate":                  0.0787,
    "defense_rate":                 0.0589,
    "recent_form":                  0.1339,
    "xg_for":                       0.0299,
    "xg_against":                   0.0299,
    "home_advantage":               0.0787,
    "capital_power":                0.0977,
    "head_to_head":                 0.0299,
    "discipline_risk":              0.0299,
    "attrition_rate":               0.0299,
    "match_interval":               0.0299,
    "injury_impact":                0.0090,
    "weather_fatigue":              0.0090,
    "travel_distance":              0.0000,
    "set_piece_conversion":         0.0000,
    "match_day_motivation":         0.0000,
    "tactical_adaptability":        0.0000,
    "expected_goals_difference":    0.0389,
    "player_availability_impact":   0.0000,
    "match_trend":                  0.0000,
    "referee_tendency":             0.0000,
    "elo":                          0.2000,  # ELO 0.20 (v8.1)
}

# ─── J1〜J3 チーム推定資本力スコア (静的DB) ────────────
# 根拠: Jリーグ開示営業費用・Transfermarkt推定スカッド価値・各社IR資料
# Peeters(2018): log(squad_value)がシーズン勝点分散の65-72%を説明
_TEAM_CAPITAL_SCORES: dict[str, float] = {
    # J1 最上位 (大企業・楽天・日産バック / 推定年俸総額 >15億円)
    "ヴィッセル神戸":           0.95,  # 楽天グループ (三木谷オーナー)
    "浦和レッズ":               0.92,  # 三菱グループ / DAZN
    "横浜F・マリノス":          0.88,  # 日産・シティグループ
    "川崎フロンターレ":         0.82,  # 富士通・UIターン効果
    "ガンバ大阪":               0.80,  # パナソニック系
    "鹿島アントラーズ":         0.78,  # メルカリ (上場IT企業)
    # J1 上位 (中堅企業バック / 10-15億円)
    "名古屋グランパス":         0.73,  # トヨタ自動車
    "FC東京":                   0.70,  # 東京ガス / MIXI
    "セレッソ大阪":             0.68,  # ヤンマー
    "サンフレッチェ広島":       0.65,
    "FC町田ゼルビア":           0.68,  # サイバーエージェント (上場IT)
    "北海道コンサドーレ札幌":   0.60,
    "柏レイソル":               0.60,  # 日立製作所
    # J1 中位 (7-10億円)
    "アビスパ福岡":             0.55,
    "京都サンガF.C.":           0.52,
    "東京ヴェルディ":           0.50,
    "アルビレックス新潟":       0.48,
    "湘南ベルマーレ":           0.45,
    # J2 上位 (5-8億円)
    "ジェフユナイテッド千葉":   0.54,
    "ベガルタ仙台":             0.50,
    "ジュビロ磐田":             0.55,  # ヤマハ
    "サガン鳥栖":               0.52,  # 林田グループ
    "V・ファーレン長崎":        0.47,  # ジャパネットたかた
    "ファジアーノ岡山":         0.44,
    "モンテディオ山形":         0.42,
    "ヴァンフォーレ甲府":       0.40,
    "大分トリニータ":           0.40,
    "ロアッソ熊本":             0.36,
    # J3 (3-5億円)
    "ギラヴァンツ北九州":       0.32,
    "レノファ山口FC":           0.32,
    "鹿児島ユナイテッドFC":     0.30,
    "FC琉球":                   0.30,
    "愛媛FC":                   0.32,
}
_DEFAULT_CAPITAL_SCORE = 0.45  # データなしチームのデフォルト値

# ─── チーム別ホームアドバンテージ (2024-2025実績) ─────────
# score = ホーム勝率 + ホーム引分率 * 0.5 (0.0〜1.0)
# データ: data.j-league.or.jp 2024+2025 計760試合
_TEAM_HOME_ADVANTAGE: dict[str, float] = {
    "鹿島アントラーズ":         0.776,
    "ガンバ大阪":               0.671,
    "サンフレッチェ広島":       0.671,
    "浦和レッズ":               0.658,
    "ヴィッセル神戸":           0.645,
    "柏レイソル":               0.605,
    "セレッソ大阪":             0.579,
    "川崎フロンターレ":         0.579,
    "FC町田ゼルビア":           0.566,
    "名古屋グランパス":         0.526,
    "FC東京":                   0.526,
    "アビスパ福岡":             0.526,
    "東京ヴェルディ":           0.513,
    "京都サンガF.C.":           0.513,
    "清水エスパルス":           0.500,
    "ファジアーノ岡山":         0.474,
    "横浜F・マリノス":          0.474,
    "北海道コンサドーレ札幌":   0.447,
    "横浜FC":                   0.421,
    "ジュビロ磐田":             0.421,
    "湘南ベルマーレ":           0.395,
    "サガン鳥栖":               0.395,
    "アルビレックス新潟":       0.355,
}
_DEFAULT_HOME_ADVANTAGE = 0.520  # J1平均

# ─── 昇格組フラグ (2026シーズン) ─────────────────────────
# J2からの昇格チームはELO蓄積が浅い＋適応期間があるためペナルティ
_PROMOTED_2026: set[str] = {
    "ジェフユナイテッド千葉",
    "V・ファーレン長崎",
    "水戸ホーリーホック",  # J2J3百年構想リーグから
}
_PROMOTED_ELO_PENALTY = 0.05  # ELOスコアから減算

# ─── 移動距離・疲労 ────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Haversine公式で2点間の距離(km)を計算
    サッカー科学: 600km以上の移動は翌日以降の試合でも累積疲労に影響
    (Drust et al., 2012; Reilly et al., 1997)
    """
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2
    return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)), 1)


def travel_fatigue_score(distance_km: float) -> float:
    """
    移動距離 → 疲労スコア (0.0〜1.0 高いほど疲労)
    Gemini設計式: travel_index = distance / 1000
    科学的根拠: 時差・気圧変化・体内時計乱れ複合効果
    """
    idx = distance_km / 1000.0
    if idx < 0.3:    return 0.00   # 300km未満: 影響なし
    elif idx < 0.6:  return 0.10   # 300-600km: 軽微
    elif idx < 1.0:  return 0.25   # 600-1000km: 中程度
    elif idx < 1.5:  return 0.50   # 1000-1500km: 大 (本州-北海道等)
    else:            return 0.75   # 1500km超: 極大 (沖縄等)


# ─── 各パラメータのスコア化 ──────────────────────────────

def score_capital_power(home_team: str, away_team: str) -> tuple[float, float]:
    """
    資本力スコア: 親会社収益・年俸総額に基づくスカッドの質 (0.0〜1.0)
    根拠: Peeters(2018) Journal of Sports Economics - log(squad_value) explains
          65-72% of variance in final season points across 5 European leagues.
    J-League: Transfermarkt推定値・各社IR資料・公開報道をもとに作成。
    """
    h = _TEAM_CAPITAL_SCORES.get(home_team, _DEFAULT_CAPITAL_SCORE)
    a = _TEAM_CAPITAL_SCORES.get(away_team, _DEFAULT_CAPITAL_SCORE)
    return h, a


def score_discipline_risk(home_cards: dict, away_cards: dict) -> tuple[float, float]:
    """
    規律リスクスコア: イエローカード累積による守備安定性 (0.0〜1.0, 高いほど安定)
    カードが多いほど: 累積停止リスク・守備陣形の崩壊・心理的プレッシャー

    Parameters
    ----------
    home_cards/away_cards: {yellow_per_game: float, red_per_game: float}
    データなし: 中立値 (0.60) を返す

    根拠: 1試合あたりのカード数と守備失点には正の相関 (Lago-Peñas 2016)
    """
    def _score(cards: dict) -> float:
        if not cards:
            return 0.60
        yell_pg = float(cards.get("yellow_per_game", 1.8))
        red_pg  = float(cards.get("red_per_game",   0.05))
        # J1平均 ~1.8枚/試合。超過分がリスク
        penalty = 0.0
        if yell_pg > 2.5:   penalty += 0.25
        elif yell_pg > 2.0: penalty += 0.15
        elif yell_pg > 1.5: penalty += 0.06
        penalty += red_pg * 3.0   # 退場は守備上の重大リスク
        return round(max(0.10, 0.78 - penalty), 3)

    return _score(home_cards), _score(away_cards)


def score_attrition_rate(
    home_injuries: list[dict],
    away_injuries: list[dict],
    squad_size: int = 25,
) -> tuple[float, float]:
    """
    損耗率スコア: 怪我人のスカッド比率による戦力維持能力 (0.0〜1.0)
    score_injury() が「1人あたりの絶対ペナルティ」を計算するのに対し、
    こちらは「スカッド全体の何%が失われているか」という深度指標。

    根拠: チームデプスは交代オプションと戦術柔軟性に直接影響
    """
    def _score(injuries: list[dict]) -> float:
        if not injuries:
            return 0.80  # 完全無傷 = 最高スコア
        ratio = len(injuries) / squad_size  # 0〜1
        # 長期離脱選手の重み付け
        long_term = sum(
            1 for inj in injuries
            if any(kw in str(inj.get("status", "")).lower()
                   for kw in ["長期", "数ヶ月", "全治", "手術", "週以上"])
        )
        base = 0.80 - ratio * 0.60      # 怪我率ペナルティ
        base -= long_term * 0.08         # 長期離脱追加ペナルティ
        return round(max(0.10, base), 3)

    return _score(home_injuries), _score(away_injuries)


def score_match_interval(home_days: int, away_days: int) -> tuple[float, float]:
    """
    試合間隔疲労スコア (U字型, 0.0〜1.0, 高いほどコンディション良好)
    根拠:
      - 中3日以内: パフォーマンス最大8%低下 (Drust et al. 2012, BJSM)
      - 中6-7日: 最適コンディション (Gemini設計)
      - 中10日超: 試合感覚低下・筋温低下 (Pope et al. 2015)
    0 = 不明 → 中立値 (0.52)

    Gemini v3推奨スコア:
      1-2日: 0.40, 3日: 0.45, 4-5日: 0.53, 6-7日: 0.55, 8-10日: 0.48, >10: 0.47
    """
    def _score(days: int) -> float:
        if days <= 0:    return 0.52   # 不明 → 中立
        elif days <= 2:  return 0.40   # 中1-2日: 高疲労 (-8%パフォーマンス)
        elif days == 3:  return 0.45   # 中3日: やや疲労 (ミッドウィーク後)
        elif days <= 5:  return 0.53   # 中4-5日: 通常コンディション
        elif days <= 7:  return 0.55   # 中6-7日: 最適 (1週間準備)
        elif days <= 10: return 0.48   # 中8-10日: 試合感覚低下
        else:            return 0.47   # 中10日超: 長期ブランク

    return _score(home_days), _score(away_days)


def score_set_piece_conversion_rate(
    home_set_pieces: dict,
    away_set_pieces: dict,
) -> tuple[float, float]:
    """
    セットプレー得点率スコア (0.0〜1.0, 高いほど良い)
    Gemini提案 v9: セットプレーからの得点能力 → 試合の流れを変える可能性を評価
    データなし: 中立値 0.5 を返す

    Parameters: {"attempts": int, "successes": int}
    """
    def _score(sp: dict) -> float:
        if not sp:
            return 0.5
        attempts  = float(sp.get("attempts",  0))
        successes = float(sp.get("successes", 0))
        if attempts == 0:
            return 0.5
        rate = successes / attempts
        return round(min(1.0, max(0.0, 0.3 + rate * 0.7)), 3)

    return _score(home_set_pieces), _score(away_set_pieces)


def score_match_day_motivation(
    home_motivation: dict,
    away_motivation: dict,
) -> tuple[float, float]:
    """
    試合当日モチベーションスコア (0.0〜1.0, 高いほど良い)
    Gemini提案 v9: 選手コメント・SNS・試合文脈から推定されるチームの気勢
    データなし: 中立値 0.5 を返す

    Parameters: {"level": float}  # 0.0〜1.0 で事前に数値化
    """
    def _score(mot: dict) -> float:
        if not mot:
            return 0.5
        level = float(mot.get("level", 0.5))
        return round(min(1.0, max(0.0, level)), 3)

    return _score(home_motivation), _score(away_motivation)


def score_team_strength(home_stats: dict, away_stats: dict) -> tuple[float, float]:
    """
    勝点・順位・得失点差に基づくチーム強度スコア (0.0〜1.0)
    2026年対応: PK勝負を考慮した勝点換算を使用。
    得失点差も補正項として加味（実力差を強調）。
    """
    h_pts = float(home_stats.get("勝点", 0))
    a_pts = float(away_stats.get("勝点", 0))

    # 得失点差を補正に使う (+1差 ≈ 0.5pt相当)
    def _gd(stats: dict) -> float:
        raw = stats.get("得失点差", "0")
        try:
            return float(str(raw).replace("+", ""))
        except ValueError:
            return 0.0

    h_effective = h_pts + _gd(home_stats) * 0.5
    a_effective = a_pts + _gd(away_stats) * 0.5
    total = max(h_effective + a_effective, 1)
    return round(h_effective / total, 3), round(a_effective / total, 3)


def score_attack_rate(home_stats: dict, away_stats: dict) -> tuple[float, float]:
    """
    1試合あたり平均得点率 (攻撃力) をスコア化 (0.0〜1.0)
    Dixon-Colesモデルの攻撃パラメータ λ に相当。
    シーズン序盤（試合数 < 6）は平均値へのシュリンクを適用。
    根拠: Goals scored rate × season points r≈0.72-0.78 (Dixon & Coles 1997)
    """
    LEAGUE_AVG = 1.3  # J1 平均得点/試合
    def _rate(stats: dict) -> float:
        games = max(float(stats.get("試合", 1)), 1)
        gf    = float(stats.get("得点", LEAGUE_AVG))
        rate  = gf / games
        # シーズン序盤は平均値にシュリンク（試合数が少ないほど強く）
        shrink = max(0.0, 1.0 - games / 10.0)
        return rate * (1 - shrink) + LEAGUE_AVG * shrink

    h_norm = min(_rate(home_stats) / (LEAGUE_AVG * 2), 1.0)
    a_norm = min(_rate(away_stats) / (LEAGUE_AVG * 2), 1.0)
    return round(h_norm, 3), round(a_norm, 3)


def score_defense_rate(home_stats: dict, away_stats: dict) -> tuple[float, float]:
    """
    1試合あたり平均失点率 (守備力) をスコア化 (0.0〜1.0)
    失点が少ないほど高スコア。シーズン序盤シュリンク適用。
    根拠: Goals conceded rate × season points r≈-0.70-0.75 (Dixon & Coles 1997)
    """
    LEAGUE_AVG = 1.3  # J1 平均失点/試合
    def _rate(stats: dict) -> float:
        games = max(float(stats.get("試合", 1)), 1)
        ga    = float(stats.get("失点", LEAGUE_AVG))
        rate  = ga / games
        shrink = max(0.0, 1.0 - games / 10.0)
        return rate * (1 - shrink) + LEAGUE_AVG * shrink

    h_norm = max(0.0, 1.0 - _rate(home_stats) / (LEAGUE_AVG * 2))
    a_norm = max(0.0, 1.0 - _rate(away_stats) / (LEAGUE_AVG * 2))
    return round(h_norm, 3), round(a_norm, 3)


def score_xg_differential(
    home_xg: dict,
    away_xg: dict,
) -> tuple[float, float]:
    """
    xG差分 (期待ゴール for - against) をスコア化 (0.0〜1.0)
    根拠: xG差分はシーズン勝点と r≈0.87-0.92 の相関 (StatsBomb, Caley 2015)
         +8〜12pp の予測精度向上が報告されている。
    データなし (J2/J3 or FBref失敗) → 中立 (0.5, 0.5) を返す。
    """
    if not home_xg or not away_xg:
        return 0.5, 0.5  # xGデータなし → 中立（貢献度=0）

    h_xgf = float(home_xg.get("xg_for",     1.3))
    h_xga = float(home_xg.get("xg_against", 1.3))
    a_xgf = float(away_xg.get("xg_for",     1.3))
    a_xga = float(away_xg.get("xg_against", 1.3))

    h_diff = h_xgf - h_xga   # positive = 攻守ともに優位
    a_diff = a_xgf - a_xga

    # 典型値域: -1.5〜+1.5 per game → 0〜1 に正規化
    h_norm = min(max((h_diff + 1.5) / 3.0, 0.0), 1.0)
    a_norm = min(max((a_diff + 1.5) / 3.0, 0.0), 1.0)
    return round(h_norm, 3), round(a_norm, 3)


def score_xg_for(
    home_xg: dict,
    away_xg: dict,
) -> tuple[float, float]:
    """期待得点(xG For)スコア (0.0〜1.0, 高いほど攻撃力がある)"""
    def _score(xg_for: float | None) -> float:
        if xg_for is None:
            return 0.50
        if xg_for <= 0.5:  return 0.10
        if xg_for <= 1.0:  return 0.10 + (xg_for - 0.5) * 0.8
        if xg_for <= 1.5:  return 0.50 + (xg_for - 1.0) * 0.5
        if xg_for <= 2.0:  return 0.75 + (xg_for - 1.5) * 0.3
        return round(min(1.0, 0.90 + (xg_for - 2.0) * 0.1), 3)
    h = None if not home_xg else home_xg.get("xg_for")
    a = None if not away_xg else away_xg.get("xg_for")
    return round(_score(float(h) if h is not None else None), 3), \
           round(_score(float(a) if a is not None else None), 3)


def score_xg_against(
    home_xg: dict,
    away_xg: dict,
) -> tuple[float, float]:
    """期待失点(xG Against)スコア (0.0〜1.0, 高いほど守備力がある)"""
    def _score(xg_against: float | None) -> float:
        if xg_against is None:
            return 0.50
        if xg_against >= 2.0:  return 0.10
        if xg_against >= 1.5:  return 0.10 + (2.0 - xg_against) * 0.8
        if xg_against >= 1.0:  return 0.50 + (1.5 - xg_against) * 0.5
        if xg_against >= 0.5:  return 0.75 + (1.0 - xg_against) * 0.3
        return round(min(1.0, 0.90 + (0.5 - xg_against) * 0.1), 3)
    h = None if not home_xg else home_xg.get("xg_against")
    a = None if not away_xg else away_xg.get("xg_against")
    return round(_score(float(h) if h is not None else None), 3), \
           round(_score(float(a) if a is not None else None), 3)


def score_tactical_adaptability(
    home_tactics: dict,
    away_tactics: dict,
) -> tuple[float, float]:
    """戦術的適応能力スコア (0.0〜1.0, 高いほど柔軟な対応が可能)"""
    def _score(td: dict) -> float:
        if not td:
            return 0.50
        score = (
            td.get("change_frequency", 0.5) * 0.30 +
            td.get("success_rate",     0.5) * 0.40 +
            td.get("formation_diversity", 0.5) * 0.20 +
            td.get("player_diversity", 0.5) * 0.10
        )
        return round(max(0.0, min(1.0, score)), 3)
    return _score(home_tactics), _score(away_tactics)


def score_recent_form(form: list[str] | str) -> float:
    """
    直近5試合フォーム → スコア (0.0〜1.0)
    入力: ['W','D','L',...] または "WWDLW" のような文字列
    W=1.0, D=0.5, L=0.0 の加重平均（直近ほど重み大）
    PK勝(p/P)→0.7, PK負(k/K)→0.3 として処理。
    """
    if isinstance(form, str):
        form = list(form.upper())
    if not form:
        return 0.5
    weights = [0.35, 0.25, 0.20, 0.12, 0.08][:len(form)]
    values = {"W": 1.0, "P": 0.7, "D": 0.5, "K": 0.3, "L": 0.0}
    total_w = sum(weights)
    return round(sum(values.get(r, 0.5) * w for r, w in zip(form, weights)) / total_w, 3)


def score_home_advantage(home_team: str = "") -> float:
    """
    チーム別ホームアドバンテージ (2024-2025実績ベース)
    home_team が空の場合はJ1平均を返す (後方互換)
    """
    if not home_team:
        return _DEFAULT_HOME_ADVANTAGE
    return _TEAM_HOME_ADVANTAGE.get(home_team, _DEFAULT_HOME_ADVANTAGE)


def score_h2h(h2h: dict, is_home: bool) -> float:
    """
    H2H成績スコア (0.0〜1.0)
    """
    total = max(h2h.get("total", 1), 1)
    wins  = h2h.get("home_wins" if is_home else "away_wins", 0)
    draws = h2h.get("draws", 0)
    return round((wins + draws * 0.5) / total, 3)


def score_injury(injuries: list[dict]) -> float:
    """
    怪我・出場停止による戦力スコア (1.0=影響なし, 0.0=主力全滅)
    選手1人あたり -0.08〜-0.15 のペナルティ
    """
    if not injuries:
        return 1.0
    penalty = min(len(injuries) * 0.10, 0.50)
    return round(1.0 - penalty, 3)


def score_weather(weather: dict) -> float:
    """
    天気コンディション → チームスコアへの影響係数
    疲労スコアが高いほど両チーム均等に影響するため中立的
    Returns: 平均影響係数 (0.5 = 中立)
    """
    fatigue = weather.get("fatigue_factor", 0.1)
    # 悪天候はホームチームが慣れている分若干有利: 0.52〜0.55
    return round(0.55 - fatigue * 0.10, 3)


def score_expected_goals_difference(home_xg: dict, away_xg: dict) -> tuple[float, float]:
    """Expected Goals Difference (xGD) スコア (0.0〜1.0, 高いほど良い)
    チャンス創出・阻止能力を評価し、拮抗した試合でのドロー予測精度を向上させる。
    データ: xg_for_per_game, xg_against_per_game を含む辞書
    """
    def _score(xg_data: dict) -> float:
        if not xg_data:
            return 0.5
        xg  = float(xg_data.get("xg_for_per_game",     xg_data.get("xg_for",     1.2)))
        xga = float(xg_data.get("xg_against_per_game",  xg_data.get("xg_against", 1.2)))
        xgd = xg - xga
        # -2.0〜+2.0 の範囲を 0.0〜1.0 に線形変換
        return round(max(0.0, min(1.0, (xgd + 2.0) / 4.0)), 3)
    return _score(home_xg), _score(away_xg)


def score_player_availability_impact(
    home_player_impact: dict, away_player_impact: dict
) -> tuple[float, float]:
    """Player Availability Impact (PAI) スコア (0.0〜1.0, 高いほど影響が少ない=良い)
    主力選手の欠場がチームに与える影響を数値化する。
    データ: total_impact_score (0.0=影響なし, 1.0=重大影響) を含む辞書
    データ未取得時はデフォルト 0.5 (中立) を返す。
    """
    def _score(impact_data: dict) -> float:
        if not impact_data:
            return 0.5
        total_impact = float(impact_data.get("total_impact_score", 0.0))
        # 0.0 (影響なし) → 1.0, 1.0 (重大影響) → 0.5, 2.0 (最大影響) → 0.0
        return round(max(0.0, min(1.0, 1.0 - (total_impact / 2.0))), 3)
    return _score(home_player_impact), _score(away_player_impact)


def score_match_trend(home_trends: dict, away_trends: dict) -> tuple[float, float]:
    """試合展開の傾向スコア (0.0〜1.0, 高いほど良い傾向)
    リードを守る力・劣勢からの追いつき力・逆転率を評価。
    データキー:
      lead_win_rate    : リードした試合での勝率 (float)
      comeback_rate    : リードを許した試合での追いつき率 (float)
      reverse_win_rate : 逆転勝ち率 (float)
      reverse_lose_rate: 逆転負け率 (float)
    データ未取得時は中立値 0.5 を返す。
    """
    def _score(trends: dict) -> float:
        if not trends:
            return 0.50
        lead_win_rate    = float(trends.get("lead_win_rate",    0.70))
        comeback_rate    = float(trends.get("comeback_rate",    0.25))
        reverse_win_rate = float(trends.get("reverse_win_rate", 0.10))
        reverse_lose_rate = float(trends.get("reverse_lose_rate", 0.15))
        score = (
            lead_win_rate    * 0.4
            + comeback_rate    * 0.3
            + reverse_win_rate * 0.2
            - reverse_lose_rate * 0.1
        )
        # 基準値 ≈ 0.36 (デフォルト); 0.2 〜 0.8 → 0.0 〜 1.0 に正規化
        return round(max(0.0, min(1.0, (score - 0.2) / 0.6)), 3)
    return _score(home_trends), _score(away_trends)


def score_referee_tendency(referee_stats: dict) -> tuple[float, float]:
    """審判の傾向スコア
    ホーム/アウェイ別 (0.0〜1.0, 高いほど当該チームに有利な傾向)。
    データキー:
      avg_yellow_cards_per_game: 平均イエローカード数/試合 (float)
      avg_red_cards_per_game   : 平均レッドカード数/試合 (float)
      home_pk_rate             : ホームにPKを与えた割合 (0.0-1.0)
      away_pk_rate             : アウェイにPKを与えた割合 (0.0-1.0)
      avg_additional_time      : 平均アディショナルタイム (float, 分)
    データ未取得時は中立値 (0.5, 0.5) を返す。
    """
    if not referee_stats:
        return 0.50, 0.50
    avg_yellow    = float(referee_stats.get("avg_yellow_cards_per_game", 3.5))
    avg_red       = float(referee_stats.get("avg_red_cards_per_game",    0.15))
    home_pk_rate  = float(referee_stats.get("home_pk_rate",  0.5))
    away_pk_rate  = float(referee_stats.get("away_pk_rate",  0.5))
    avg_add_time  = float(referee_stats.get("avg_additional_time", 8.0))

    # カードペナルティ: 多いほど試合が荒れる → ベーススコアを下げる
    card_penalty = (avg_yellow * 0.05) + (avg_red * 0.5)
    card_score   = max(0.0, 0.7 - card_penalty)

    # アディショナルタイム: 平均(8分)から乖離するほど不確実性が増す
    add_time_impact = 0.0
    if avg_add_time > 10.0:
        add_time_impact = -0.05
    elif avg_add_time < 6.0:
        add_time_impact = -0.03

    base_score = card_score * 0.6 + 0.5 * 0.4 + add_time_impact
    normalized  = max(0.0, min(1.0, (base_score - 0.3) / 0.6))

    # PKバイアス: ホームPK率が高いほどホーム有利
    pk_bias = (home_pk_rate - away_pk_rate) * 0.2
    home_score = round(max(0.0, min(1.0, normalized + pk_bias)), 3)
    away_score = round(max(0.0, min(1.0, normalized - pk_bias)), 3)
    return home_score, away_score


# ─── パラメータ貢献度計算 ───────────────────────────────

def calculate_parameter_contributions(
    home_team: str,
    away_team: str,
    home_stats: dict,
    away_stats: dict,
    home_form: list[str],
    away_form: list[str],
    h2h: dict,
    weather: dict,
    home_injuries: list[dict],
    away_injuries: list[dict],
    home_venue: dict,
    away_venue: dict,
    home_xg: dict | None = None,          # FBref xG (J1のみ)
    away_xg: dict | None = None,
    home_cards: dict | None = None,       # カード規律統計
    away_cards: dict | None = None,
    home_days: int = 0,                   # 試合間隔 (日数)
    away_days: int = 0,
    home_set_pieces: dict | None = None,       # セットプレー統計
    away_set_pieces: dict | None = None,
    home_motivation: dict | None = None,       # 試合当日モチベーション
    away_motivation: dict | None = None,
    home_tactics: dict | None = None,          # 戦術的適応能力
    away_tactics: dict | None = None,
    home_player_impact: dict | None = None,    # PAI: 主力選手欠場影響
    away_player_impact: dict | None = None,
    home_match_trends: dict | None = None,     # 試合展開傾向統計
    away_match_trends: dict | None = None,
    referee_stats: dict | None = None,         # 審判傾向統計
    elo_home_score: float | None = None,       # ELOホーム期待勝率 (0-1)
    elo_away_score: float | None = None,       # ELOアウェイ期待勝率 (0-1)
) -> dict[str, Any]:
    """
    各パラメータのホーム有利度スコアと重み付き貢献度を計算。

    Returns
    -------
    {
      "parameters": {
        param_name: {
          "home_score": float,
          "away_score": float,
          "home_advantage": float,  # home - away
          "weight": float,
          "contribution": float,    # home_advantage * weight
        }
      },
      "raw_home_advantage": float,  # 加重合計
      "distance_km": float,
    }
    """
    # 移動距離
    dist_km = haversine_km(
        away_venue["lat"], away_venue["lon"],
        home_venue["lat"], home_venue["lon"],
    )
    travel_fat = travel_fatigue_score(dist_km)

    # 各スコア
    h_str, a_str        = score_team_strength(home_stats, away_stats)
    h_atk, a_atk        = score_attack_rate(home_stats, away_stats)
    h_def, a_def        = score_defense_rate(home_stats, away_stats)
    h_form              = score_recent_form(home_form)
    a_form              = score_recent_form(away_form)
    h_xg_s, a_xg_s     = score_xg_differential(home_xg or {}, away_xg or {})
    h_xg_for, a_xg_for      = score_xg_for(home_xg or {}, away_xg or {})
    h_xg_against, a_xg_against = score_xg_against(home_xg or {}, away_xg or {})
    h_home              = score_home_advantage(home_team)
    a_away              = 1.0 - h_home
    h_cap, a_cap        = score_capital_power(home_team, away_team)
    h_h2h               = score_h2h(h2h, is_home=True)
    a_h2h               = score_h2h(h2h, is_home=False)
    h_disc, a_disc      = score_discipline_risk(home_cards or {}, away_cards or {})
    h_att, a_att        = score_attrition_rate(home_injuries, away_injuries)
    h_interval, a_interval = score_match_interval(home_days, away_days)
    h_inj               = score_injury(home_injuries)
    a_inj               = score_injury(away_injuries)
    h_weather           = score_weather(weather)
    a_weather           = 1.0 - h_weather + 0.45
    h_travel            = 1.0
    a_travel            = 1.0 - travel_fat
    h_setp, a_setp      = score_set_piece_conversion_rate(
                              home_set_pieces or {}, away_set_pieces or {}
                          )
    h_motiv, a_motiv    = score_match_day_motivation(
                              home_motivation or {}, away_motivation or {}
                          )
    h_tact, a_tact      = score_tactical_adaptability(
                              home_tactics or {}, away_tactics or {}
                          )
    h_xgd, a_xgd       = score_expected_goals_difference(
                              home_xg or {}, away_xg or {}
                          )
    h_pai, a_pai        = score_player_availability_impact(
                              home_player_impact or {}, away_player_impact or {}
                          )
    h_trend, a_trend    = score_match_trend(
                              home_match_trends or {}, away_match_trends or {}
                          )
    h_ref, a_ref        = score_referee_tendency(referee_stats or {})

    params: dict[str, dict] = {}
    raw_adv = 0.0

    for name, h_score, a_score in [
        ("team_strength",   h_str,      a_str),
        ("attack_rate",     h_atk,      a_atk),
        ("defense_rate",    h_def,      a_def),
        ("recent_form",     h_form,     a_form),
        ("xg_for",          h_xg_for,     a_xg_for),
        ("xg_against",      h_xg_against, a_xg_against),
        ("home_advantage",  h_home,     a_away),
        ("capital_power",   h_cap,      a_cap),
        ("head_to_head",    h_h2h,      a_h2h),
        ("discipline_risk", h_disc,     a_disc),
        ("attrition_rate",  h_att,      a_att),
        ("match_interval",  h_interval, a_interval),
        ("injury_impact",   h_inj,      a_inj),
        ("weather_fatigue",        h_weather,  a_weather),
        ("travel_distance",        h_travel,   a_travel),
        ("set_piece_conversion",        h_setp,  a_setp),
        ("match_day_motivation",        h_motiv, a_motiv),
        ("tactical_adaptability",       h_tact,  a_tact),
        ("expected_goals_difference",   h_xgd,   a_xgd),
        ("player_availability_impact",  h_pai,   a_pai),
        ("match_trend",                 h_trend, a_trend),
        ("referee_tendency",            h_ref,   a_ref),
        ("elo",
            max(0.0, (elo_home_score if elo_home_score is not None else 0.5)
                - (_PROMOTED_ELO_PENALTY if home_team in _PROMOTED_2026 else 0.0)),
            max(0.0, (elo_away_score if elo_away_score is not None else 0.5)
                - (_PROMOTED_ELO_PENALTY if away_team in _PROMOTED_2026 else 0.0)),
        ),
    ]:
        adv = round(h_score - a_score, 3)
        w = MODEL_WEIGHTS.get(name, 0.0)
        contrib = round(adv * w, 4)
        raw_adv += contrib
        params[name] = {
            "home_score": h_score,
            "away_score": a_score,
            "home_advantage": adv,
            "weight": w,
            "contribution": contrib,
        }

    # 接近度: raw_adv が 0 に近いほど高い (drawの信号として使用)
    closeness = round(max(0.0, 1.0 - abs(raw_adv) * 3.0), 4)

    return {
        "parameters": params,
        "raw_home_advantage": round(raw_adv, 4),
        "closeness": closeness,
        "distance_km":  dist_km,
        "travel_fatigue": travel_fat,
        "home_days":    home_days,
        "away_days":    away_days,
        "capital_home": _TEAM_CAPITAL_SCORES.get(home_team, _DEFAULT_CAPITAL_SCORE),
        "capital_away": _TEAM_CAPITAL_SCORES.get(away_team, _DEFAULT_CAPITAL_SCORE),
    }


# ─── 確率変換 ────────────────────────────────────────────

def _softmax3(lh: float, ld: float, la: float) -> tuple[float, float, float]:
    """数値安定な3クラスsoftmax"""
    m = max(lh, ld, la)
    eh = math.exp(lh - m)
    ed = math.exp(ld - m)
    ea = math.exp(la - m)
    s = eh + ed + ea
    return eh / s, ed / s, ea / s


# 3ロジット変換のパラメータ (v7再探索済み, val=2025 F1=0.435)
# 3ロジット変換パラメータ - Primary (v7 refined)
_3LOGIT_PARAMS = {
    "scale_ha":   1.44,    # 勝敗方向の感度 (v7)
    "bias_home":  0.14,    # ホームバイアス (v7)
    "bias_away":  0.07,    # アウェイバイアス (v7)
    "scale_draw": 0.80,    # draw感度 (v7)
    "bias_draw":  -0.60,   # drawベースロジット (v7)
}

# 3ロジット変換パラメータ - Shadow v8.1 (比較用)
V8_1_3LOGIT_PARAMS = {
    "scale_ha":   1.70,
    "bias_home":  0.12,
    "bias_away":  0.07,
    "scale_draw": 1.10,
    "bias_draw":  -0.90,
}


def advantage_to_probs(
    raw_advantage: float,
    closeness: float = 0.5,
    mode: str = "3logit",
) -> tuple[int, int, int]:
    """
    raw_advantage (-1〜+1) + closeness (0〜1) → (home_win%, draw%, away_win%)

    Parameters
    ----------
    raw_advantage : 重み付き特徴量のホーム優勢度 (正=ホーム有利)
    closeness : 実力接近度 (1.0=完全均衡, 0.0=一方的差)
                calculate_parameter_contributions() の戻り値に含まれる
    mode : "3logit" (新方式) or "legacy" (旧シグモイド方式)

    3ロジット方式:
      logit_home = scale_ha * raw_adv + bias_home
      logit_away = -scale_ha * raw_adv + bias_away
      logit_draw = scale_draw * closeness + bias_draw
      → softmax([logit_home, logit_draw, logit_away])

    drawに独立したロジットを持たせることで、
    実力接近時にdrawがargmaxで選ばれることが可能。
    """
    if mode == "legacy":
        return _legacy_advantage_to_probs(raw_advantage)

    # mode="v8.1": shadow model
    if mode == "v8.1":
        p = V8_1_3LOGIT_PARAMS
    else:
        p = _3LOGIT_PARAMS
    logit_h = p["scale_ha"] * raw_advantage + p["bias_home"]
    logit_a = -p["scale_ha"] * raw_advantage + p["bias_away"]
    logit_d = p["scale_draw"] * closeness + p["bias_draw"]

    h, d, a = _softmax3(logit_h, logit_d, logit_a)

    h_pct = round(h * 100)
    a_pct = round(a * 100)
    d_pct = 100 - h_pct - a_pct

    return h_pct, d_pct, a_pct


def _legacy_advantage_to_probs(raw_advantage: float) -> tuple[int, int, int]:
    """旧方式 (シグモイド + draw残差)。rollback用に保持。"""
    base_home = 0.40
    base_draw = 0.25
    base_away = 0.35
    shift = math.tanh(raw_advantage * 3) * 0.30
    h = max(0.05, min(0.90, base_home + shift))
    a = max(0.05, min(0.90, base_away - shift))
    d = max(0.05, min(0.35, 1.0 - h - a))
    total = h + d + a
    h_pct = round(h / total * 100)
    a_pct = round(a / total * 100)
    d_pct = 100 - h_pct - a_pct
    return h_pct, d_pct, a_pct


# ─── Gemini 2.5 Flash 統合予測 ──────────────────────────

def compute_shadow_v8_1(
    home_team: str, away_team: str,
    home_stats: dict, away_stats: dict,
    home_form: list[str], away_form: list[str],
    elo_home_score: float | None = None,
    elo_away_score: float | None = None,
) -> dict:
    """
    Shadow model v8.1 による予測を計算する。
    Primary (v7) とは独立にraw_advantageとclosenessを計算し、
    V8_1_3LOGIT_PARAMS を使って確率に変換する。

    Returns: {"home_win_prob", "draw_prob", "away_win_prob",
              "raw_home_advantage", "closeness", "model_version"}
    """
    # v8.1の重みを一時的に使ってraw_advantageを再計算
    h_str, a_str = score_team_strength(home_stats, away_stats)
    h_atk, a_atk = score_attack_rate(home_stats, away_stats)
    h_def, a_def = score_defense_rate(home_stats, away_stats)
    h_form_s = score_recent_form(home_form)
    a_form_s = score_recent_form(away_form)
    h_ha = score_home_advantage(home_team)
    a_ha = 1.0 - h_ha
    h_cap, a_cap = score_capital_power(home_team, away_team)

    # ELO
    if elo_home_score is None:
        elo_home_score = 0.5
    if elo_away_score is None:
        elo_away_score = 0.5
    # 昇格組ペナルティ
    if home_team in _PROMOTED_2026:
        elo_home_score = max(0.0, elo_home_score - _PROMOTED_ELO_PENALTY)
    if away_team in _PROMOTED_2026:
        elo_away_score = max(0.0, elo_away_score - _PROMOTED_ELO_PENALTY)

    W = V8_1_MODEL_WEIGHTS
    raw_adv = (
        (h_str - a_str) * W.get("team_strength", 0)
        + (h_atk - a_atk) * W.get("attack_rate", 0)
        + (h_def - a_def) * W.get("defense_rate", 0)
        + (h_form_s - a_form_s) * W.get("recent_form", 0)
        + (h_ha - a_ha) * W.get("home_advantage", 0)
        + (h_cap - a_cap) * W.get("capital_power", 0)
        + (elo_home_score - elo_away_score) * W.get("elo", 0)
    )
    closeness = max(0.0, 1.0 - abs(raw_adv) * 3.0)

    h_pct, d_pct, a_pct = advantage_to_probs(raw_adv, closeness, mode="v8.1")

    return {
        "home_win_prob": h_pct,
        "draw_prob": d_pct,
        "away_win_prob": a_pct,
        "raw_home_advantage": round(raw_adv, 4),
        "closeness": round(closeness, 4),
        "model_version": "v8.1_shadow",
    }


def compute_hybrid_v9(
    home_team: str, away_team: str,
    home_stats: dict, away_stats: dict,
    home_form: list[str], away_form: list[str],
    v7_prediction: dict,
    elo_home_score: float | None = None,
    elo_away_score: float | None = None,
    xg_home: dict | None = None,
    xg_away: dict | None = None,
) -> dict:
    """
    Hybrid v9: v7 と Skellam dynamic を動的選択する統合モデル。

    選択ルール:
    - v7のdraw警戒時 (draw>=25% かつ |home-away|<10pp) → v7 採用
    - Skellamが高確信 (max>=50%) かつ非draw → Skellam 採用
    - それ以外 → v7とSkellamの重み付き平均

    Parameters
    ----------
    v7_prediction : v7による予測結果 (home_win_prob, draw_prob, away_win_prob を含む)
    その他: Skellam計算用の入力

    Returns
    -------
    {home_win_prob, draw_prob, away_win_prob, selection, model_version}
    """
    try:
        from scripts.skellam_model import predict_skellam_dynamic
    except ImportError:
        from skellam_model import predict_skellam_dynamic

    # Skellam dynamic予測
    sk = predict_skellam_dynamic(
        home_stats, away_stats,
        elo_home_score=elo_home_score,
        elo_away_score=elo_away_score,
        xg_home=xg_home, xg_away=xg_away,
    )

    # 確率抽出
    v7_h = int(v7_prediction.get("home_win_prob", 40))
    v7_d = int(v7_prediction.get("draw_prob", 25))
    v7_a = int(v7_prediction.get("away_win_prob", 35))
    sk_h = sk["home_win_prob"]
    sk_d = sk["draw_prob"]
    sk_a = sk["away_win_prob"]

    # 選択ロジック
    v7_draw_alert = v7_d >= 25 and abs(v7_h - v7_a) < 10
    sk_max = max(sk_h, sk_d, sk_a)
    sk_argmax = "home" if sk_h == sk_max else ("draw" if sk_d == sk_max else "away")
    sk_high_conf_nondraw = sk_max >= 50 and sk_argmax != "draw"

    if v7_draw_alert:
        h, d, a = v7_h, v7_d, v7_a
        selection = "v7"
    elif sk_high_conf_nondraw:
        h, d, a = sk_h, sk_d, sk_a
        selection = "skellam"
    else:
        # 重み付き平均 (0.5/0.5)
        h = round((v7_h + sk_h) / 2)
        d = round((v7_d + sk_d) / 2)
        a = 100 - h - d
        selection = "weighted"

    return {
        "home_win_prob": h,
        "draw_prob": d,
        "away_win_prob": a,
        "predicted_score": v7_prediction.get("predicted_score", "?-?"),
        "selection": selection,
        "skellam_raw": {"home": sk_h, "draw": sk_d, "away": sk_a},
        "skellam_boost": sk.get("dynamic_boost", 0.0),
        "model_version": "hybrid_v9",
    }


def _build_prior_text(contributions: dict, home_team: str, away_team: str) -> str:
    """統計モデル事前確率テキストを生成 (Geminiプロンプト用)"""
    raw_adv = contributions.get("raw_home_advantage", 0.0)
    closeness = contributions.get("closeness", 0.5)
    h_pct, d_pct, a_pct = advantage_to_probs(raw_adv, closeness)

    # ELO情報 (あれば)
    elo_param = contributions.get("parameters", {}).get("elo", {})
    elo_text = ""
    if elo_param and elo_param.get("weight", 0) > 0:
        h_elo_s = elo_param.get("home_score", 0.5)
        a_elo_s = elo_param.get("away_score", 0.5)
        elo_text = f"\nELO期待勝率: {home_team} {h_elo_s*100:.0f}% / {away_team} {a_elo_s*100:.0f}%"

    # draw注意シグナル
    draw_signal = ""
    if closeness >= 0.5 and d_pct >= 25:
        draw_signal = f"\n※ 統計モデルは引き分けの可能性を示唆しています (接近度{closeness:.2f})"

    return (
        f"22パラメータ+ELOの重み付き分析に基づく事前予測:\n"
        f"  ホーム勝利: {h_pct}%\n"
        f"  引き分け:   {d_pct}%\n"
        f"  アウェイ勝利: {a_pct}%\n"
        f"実力接近度: {closeness:.2f} (1.0=完全均衡, 0.0=一方的)"
        f"{elo_text}"
        f"{draw_signal}"
    )


def predict_with_gemini(
    home_team: str,
    away_team: str,
    contributions: dict[str, Any],
    home_stats: dict,
    away_stats: dict,
    home_form: list[str],
    away_form: list[str],
    h2h: dict,
    weather: dict,
    home_xg: dict | None = None,
    away_xg: dict | None = None,
    home_days: int = 0,
    away_days: int = 0,
    home_cards: dict | None = None,       # 生データ: {yellow_per_game, red_per_game}
    away_cards: dict | None = None,
    home_injuries: list | None = None,    # 生データ: 怪我人リスト
    away_injuries: list | None = None,
) -> dict[str, Any]:
    """
    パラメータ貢献度をGemini 2.5 Flashに渡し、
    最終的な確率と深い分析根拠を取得する。
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        # Gemini未設定時は統計モデルのみで予測
        h_pct, d_pct, a_pct = advantage_to_probs(
            contributions["raw_home_advantage"],
            contributions.get("closeness", 0.5),
        )
        return _statistical_result(home_team, away_team, h_pct, d_pct, a_pct, contributions)

    try:
        from google import genai
        from google.genai import types as gtypes

        client = genai.Client(
            api_key=api_key,
            http_options={"timeout": 300000},  # 300秒 (ms単位)
        )

        params_text = "\n".join(
            f"  - {k}: ホーム={v['home_score']:.3f} アウェー={v['away_score']:.3f} "
            f"差={v['home_advantage']:+.3f} 重み={v['weight']} 貢献度={v['contribution']:+.4f}"
            for k, v in contributions["parameters"].items()
        )

        # 攻撃・守備率計算
        h_games = max(float(home_stats.get("試合", 1)), 1)
        a_games = max(float(away_stats.get("試合", 1)), 1)
        h_atk_r = round(float(home_stats.get("得点", 0)) / h_games, 2)
        h_def_r = round(float(home_stats.get("失点", 0)) / h_games, 2)
        a_atk_r = round(float(away_stats.get("得点", 0)) / a_games, 2)
        a_def_r = round(float(away_stats.get("失点", 0)) / a_games, 2)

        # xG テキスト
        home_xg = home_xg or {}
        away_xg = away_xg or {}
        xg_text = ""
        if home_xg or away_xg:
            xg_text = (
                f"\n## xG (期待ゴール) [FBref J1データ]\n"
                f"{home_team}: xG={home_xg.get('xg_for','N/A')}/試合, "
                f"xGA={home_xg.get('xg_against','N/A')}/試合, "
                f"SoT={home_xg.get('sot_for','N/A')}/試合\n"
                f"{away_team}: xG={away_xg.get('xg_for','N/A')}/試合, "
                f"xGA={away_xg.get('xg_against','N/A')}/試合, "
                f"SoT={away_xg.get('sot_for','N/A')}/試合"
            )

        # 資本力差・新指標テキスト
        h_cap = _TEAM_CAPITAL_SCORES.get(home_team, _DEFAULT_CAPITAL_SCORE)
        a_cap = _TEAM_CAPITAL_SCORES.get(away_team, _DEFAULT_CAPITAL_SCORE)
        capital_diff = h_cap - a_cap
        # ジャイアントキリング判定 (Gemini設計: capital_diff > 0.3 = 資本格差あり)
        gk_label = ""
        if abs(capital_diff) >= 0.30:
            richer = home_team if capital_diff > 0 else away_team
            poorer = away_team if capital_diff > 0 else home_team
            gk_label = f"⚠️ 資本格差試合: {richer}(資本力{max(h_cap,a_cap):.2f}) vs {poorer}(資本力{min(h_cap,a_cap):.2f}) → ジャイアントキリング要注意"

        # 試合間隔テキスト
        interval_text = (
            f"{home_team}: 前試合から{home_days}日 | "
            f"{away_team}: 前試合から{away_days}日"
        ) if (home_days > 0 or away_days > 0) else "試合間隔データなし"

        # カード規律テキスト（生データ + スコア）
        home_cards = home_cards or {}
        away_cards = away_cards or {}
        p_disc = contributions["parameters"].get("discipline_risk", {})
        h_yell = home_cards.get("yellow_per_game", "?")
        h_red  = home_cards.get("red_per_game",   "?")
        a_yell = away_cards.get("yellow_per_game", "?")
        a_red  = away_cards.get("red_per_game",   "?")
        disc_text = (
            f"{home_team}: イエロー{h_yell}枚/試合 レッド{h_red}枚/試合 "
            f"→ 規律スコア{p_disc.get('home_score', 0.6):.3f}\n"
            f"{away_team}: イエロー{a_yell}枚/試合 レッド{a_red}枚/試合 "
            f"→ 規律スコア{p_disc.get('away_score', 0.6):.3f}"
        )

        # 怪我人テキスト（生データ + スコア）
        home_injuries = home_injuries or []
        away_injuries = away_injuries or []
        p_att = contributions["parameters"].get("attrition_rate", {})
        h_inj_names = [inj.get("player", "不明") for inj in home_injuries[:5]]
        a_inj_names = [inj.get("player", "不明") for inj in away_injuries[:5]]
        inj_text = (
            f"{home_team}: 怪我人{len(home_injuries)}名/25スカッド"
            f"{' (' + ', '.join(h_inj_names) + ')' if h_inj_names else ''}"
            f" → 損耗スコア{p_att.get('home_score', 0.8):.2f}\n"
            f"{away_team}: 怪我人{len(away_injuries)}名/25スカッド"
            f"{' (' + ', '.join(a_inj_names) + ')' if a_inj_names else ''}"
            f" → 損耗スコア{p_att.get('away_score', 0.8):.2f}"
        )

        prompt = f"""あなたはJリーグ試合予測の専門AIアナリストです。
以下の詳細な科学的パラメータ分析を基に、試合結果を予測してください。

## 試合
ホーム: {home_team} (順位{home_stats.get('順位','?')}位, 勝点{home_stats.get('勝点','?')}, 資本力スコア{h_cap:.2f})
アウェー: {away_team} (順位{away_stats.get('順位','?')}位, 勝点{away_stats.get('勝点','?')}, 資本力スコア{a_cap:.2f})
{gk_label}

## 攻撃・守備力 (Dixon-Colesパラメータ)
{home_team}: 得点率={h_atk_r}/試合, 失点率={h_def_r}/試合
{away_team}: 得点率={a_atk_r}/試合, 失点率={a_def_r}/試合
{xg_text}

## コンディション指標 (v3新パラメータ — 生データ付き)
### 試合間隔・休息
{interval_text}

### イエローカード累積・規律リスク
{disc_text}

### 怪我人・損耗率
{inj_text}

### 資本力 (親会社収益・年俸総額)
{home_team}: 資本力スコア{h_cap:.2f} (J平均=0.45〜0.55)
{away_team}: 資本力スコア{a_cap:.2f} (J平均=0.45〜0.55)
資本差: {capital_diff:+.2f}

## 重み付きパラメータ貢献度（v3 Gemini設計モデル）
{params_text}

加重合計ホームアドバンテージスコア: {contributions['raw_home_advantage']:+.4f}
移動距離: {contributions['distance_km']}km (疲労係数: {contributions['travel_fatigue']:.2f})

## 統計モデル事前確率 (v7 3ロジット方式)
{_build_prior_text(contributions, home_team, away_team)}

## 直近フォーム
{home_team}: {' '.join(home_form)}
{away_team}: {' '.join(away_form)}

## H2H (過去{h2h.get('total',0)}試合)
{home_team} {h2h.get('home_wins',0)}勝 / 引分 {h2h.get('draws',0)} / {away_team} {h2h.get('away_wins',0)}勝

## 天気
{weather.get('description','?')} 気温{weather.get('temp_avg','?')}°C 降水{weather.get('precipitation','?')}mm

## 指示
上記データを統合分析し、以下のJSON形式のみで回答してください。
資本力格差がある場合は「ジャイアントキリング確率」を特に精密に算出すること。
上記の統計モデル事前確率は22パラメータ+ELOの重み付き分析に基づく参考値です。
この事前確率を出発点として、直近フォーム・H2H・天気・怪我等の定性情報で補正してください。
統計モデルが引き分けを示唆している場合は、その可能性を軽視しないでください。

{{
  "home_win_prob": <0-100整数>,
  "draw_prob": <0-100整数>,
  "away_win_prob": <0-100整数>,
  "predicted_score": "<H>-<A>",
  "confidence": "<high|medium|low>",
  "reasoning": "<400字以上の日本語分析。資本力差・試合間隔・規律リスク・損耗率を含む全パラメータの寄与を数値付きで論述>",
  "key_factors": ["<要因1>", "<要因2>", "<要因3>", "<要因4>", "<要因5>"],
  "upset_risk": <番狂わせリスク 0-100>,
  "giant_killing_prob": <資本力劣位チームが勝つ確率 0-100。格差なし場合はnull>,
  "model_notes": "<v3新指標（資本力・規律・損耗率・試合間隔）が予測に与えた影響>"
}}

home_win_prob + draw_prob + away_win_prob = 100 を厳守。"""

        # ストリーミングで受信（read_timeout を回避）
        _chunks: list[str] = []
        for _chunk in client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=prompt,
            config=gtypes.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.25,
                max_output_tokens=2048,
                thinking_config=gtypes.ThinkingConfig(thinking_budget=0),
            ),
        ):
            if _chunk.text:
                _chunks.append(_chunk.text)
        raw = "".join(_chunks).strip()
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip().split("```")[0]

        result = json.loads(raw)
        result["model"] = "gemini-2.5-flash"
        result["distance_km"] = contributions["distance_km"]
        result["contributions"] = contributions["parameters"]
        _normalize(result)
        return result

    except Exception as exc:
        logger.error("Gemini prediction error: %s", exc)
        h_pct, d_pct, a_pct = advantage_to_probs(
            contributions["raw_home_advantage"],
            contributions.get("closeness", 0.5),
        )
        return _statistical_result(home_team, away_team, h_pct, d_pct, a_pct, contributions)


def _normalize(r: dict) -> None:
    h = int(r.get("home_win_prob", 40))
    d = int(r.get("draw_prob", 25))
    a = int(r.get("away_win_prob", 35))
    t = h + d + a
    if t != 100 and t > 0:
        r["home_win_prob"] = round(h / t * 100)
        r["draw_prob"] = round(d / t * 100)
        r["away_win_prob"] = 100 - r["home_win_prob"] - r["draw_prob"]


def _statistical_result(
    home: str, away: str,
    h_pct: int, d_pct: int, a_pct: int,
    contributions: dict,
) -> dict:
    return {
        "home_win_prob": h_pct,
        "draw_prob": d_pct,
        "away_win_prob": a_pct,
        "predicted_score": "1-1",
        "confidence": "medium",
        "reasoning": (
            f"統計モデル予測（Gemini未使用）\n"
            f"加重ホームアドバンテージスコア: {contributions['raw_home_advantage']:+.4f}\n"
            f"移動距離: {contributions['distance_km']}km\n\n"
            "各パラメータの重み付き合計からシグモイド変換で確率を算出しています。"
        ),
        "key_factors": [
            f"チーム強度差: {contributions['parameters']['team_strength']['home_advantage']:+.3f}",
            f"資本力差: {contributions['parameters'].get('capital_power',{}).get('home_advantage',0):+.3f}",
            f"攻撃率差: {contributions['parameters']['attack_rate']['home_advantage']:+.3f}",
            f"守備率差: {contributions['parameters']['defense_rate']['home_advantage']:+.3f}",
            f"フォーム差: {contributions['parameters']['recent_form']['home_advantage']:+.3f}",
            f"xG差分: {contributions['parameters'].get('xg_differential',{}).get('home_advantage',0):+.3f}",
            f"規律リスク差: {contributions['parameters'].get('discipline_risk',{}).get('home_advantage',0):+.3f}",
            f"試合間隔: H{contributions.get('home_days',0)}日/A{contributions.get('away_days',0)}日",
        ],
        "giant_killing_prob": None,
        "upset_risk": 30,
        "model_notes": "Gemini API未設定のため統計モデルのみで予測",
        "model": "statistical-only",
        "distance_km": contributions["distance_km"],
        "contributions": contributions["parameters"],
    }


# ─── ELO レーティングシステム ──────────────────────────

class EloSystem:
    """
    試合結果から動的に更新するELOレーティング。
    バックテストと本番で同一実装を使用する。

    Usage:
        elo = EloSystem(k=32.0, home_bonus=50.0)
        for match in past_results:
            elo.update(match["home"], match["away"], match["winner"])
        h_exp, a_exp = elo.score_pair("チームA", "チームB")
    """

    def __init__(
        self,
        k: float = 32.0,
        initial: float = 1500.0,
        home_bonus: float = 50.0,
    ):
        self.k = k
        self.initial = initial
        self.home_bonus = home_bonus
        self.ratings: dict[str, float] = {}

    def get(self, team: str) -> float:
        """チームのELOレーティングを取得 (未登録=initial)"""
        return self.ratings.get(team, self.initial)

    def expected(self, ra: float, rb: float) -> float:
        """ELO期待勝率"""
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def update(self, home: str, away: str, winner: str) -> None:
        """試合結果でELOを更新"""
        rh = self.get(home) + self.home_bonus
        ra = self.get(away)
        eh = self.expected(rh, ra)
        sh = 1.0 if winner == "home" else (0.5 if winner == "draw" else 0.0)
        self.ratings[home] = self.get(home) + self.k * (sh - eh)
        self.ratings[away] = self.get(away) + self.k * ((1.0 - sh) - (1.0 - eh))

    def score_pair(self, home: str, away: str) -> tuple[float, float]:
        """ホーム/アウェイのELO期待勝率 (0-1) を返す"""
        rh = self.get(home) + self.home_bonus
        ra = self.get(away)
        eh = self.expected(rh, ra)
        return eh, 1.0 - eh
