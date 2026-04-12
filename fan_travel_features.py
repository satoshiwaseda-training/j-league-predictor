"""
fan_travel_features.py - 移動距離・ファン規模の特徴量生成

責務:
  A. 選手移動疲労 (away_travel_distance_km, travel_time_proxy, long_trip, rest days)
  B. ファン移動しやすさ (away_fan_access_penalty, weekday, late_kickoff, derby relief)
  C. ファン規模 proxy (rolling attendance, fill_rate, fanbase, core_support, vote_engagement)

設計方針:
  - 選手疲労とファン動員効果を明確に分離
  - fanbase は厳密人数ではなく proxy (0-1 スケール)
  - 欠損は None のまま保持
  - match_id をキーに結合
  - 後段で SNS/投票系 proxy を追加可能な設計

保存: data/fan_travel_features.csv
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, date as _date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
FAN_TRAVEL_CSV = DATA_DIR / "fan_travel_features.csv"


# ─── 定数 ────────────────────────────────────────────���───

LONG_TRIP_KM = 600           # 長距離移動判定
VERY_LONG_TRIP_KM = 1000     # 超長距離
CONGESTED_DAYS = 3           # 連戦判定
TRAVEL_SPEED_KMH = 250       # 移動速度 proxy (新幹線+乗換え平均)
FAN_TRAVEL_SPEED_KMH = 150   # ファンの平均移動速度 (新幹線+在来+徒歩)

# デーゲーム/ナイター閾値
DAY_GAME_BEFORE = 16
NIGHT_GAME_AFTER = 18
LATE_KICKOFF_HOUR = 19       # 遅めのキックオフ


# ─── ファンベース proxy (静的DB) ────────────────────────────
# 根拠: Jリーグ公式入場者数統計 (2022-2024 平均) + SNSフォロワー数
# スケール: 0.0 (最小) 〜 1.0 (最大)
# 更新頻度: シーズン初めに一度

_CLUB_FANBASE_PROXY: dict[str, float] = {
    # J1 大規模 (平均動員 25,000+, SNS 100万+)
    "浦和レッズ":               0.98,
    "横浜F・マリノス":          0.82,
    "ヴィッセル神戸":           0.80,
    "FC東京":                   0.78,
    "ガンバ大阪":               0.75,
    "鹿島アントラーズ":         0.73,
    "名古屋グランパス":         0.72,
    "川崎フロンターレ":         0.70,
    # J1 中規模 (平均動員 15,000-25,000)
    "セレッソ大阪":             0.65,
    "サンフレッチェ広島":       0.63,
    "アルビレックス新潟":       0.62,
    "北海道コンサドーレ札幌":   0.60,
    "清水エスパルス":           0.58,
    "柏レイソル":               0.55,
    "FC町田ゼルビア":           0.50,
    "東京ヴェルディ":           0.48,
    "アビスパ福岡":             0.47,
    "京都サンガF.C.":           0.45,
    "湘南ベルマーレ":           0.42,
    # J2 / J1 昇格組
    "ベガルタ仙台":             0.55,
    "ジュビロ磐田":             0.52,
    "ジェフユナイテッド千葉":   0.50,
    "大分トリニータ":           0.45,
    "V・ファーレン長崎":        0.44,
    "ファジアーノ岡山":         0.43,
    "モンテディオ山形":         0.40,
    "サガン鳥栖":               0.38,
    "ヴァンフォーレ甲府":       0.36,
    "ロアッソ熊本":             0.35,
    "ギラヴァンツ北九州":       0.30,
    "レノファ山口FC":           0.28,
    "鹿児島ユナイテッドFC":     0.26,
    "FC琉球":                   0.24,
    "愛媛FC":                   0.25,
    "栃木SC":                   0.30,
}
_DEFAULT_FANBASE = 0.35


# ─── コアサポート proxy ──────────────────────────────────
# シーズンチケット販売規模 + サポーター団体数の推定
# 0.0〜1.0

_CLUB_CORE_SUPPORT: dict[str, float] = {
    "浦和レッズ":               0.95,  # 北ゴール裏 20,000+
    "横浜F・マリノス":          0.72,
    "鹿島アントラーズ":         0.70,
    "ヴィッセル神戸":           0.68,
    "ガンバ大阪":               0.67,
    "FC東京":                   0.65,
    "川崎フロンターレ":         0.65,
    "名古屋グランパス":         0.62,
    "セレッソ大阪":             0.58,
    "サンフレッチェ広島":       0.56,
    "アルビレックス新潟":       0.55,
    "清水エスパルス":           0.53,
    "北海道コンサドーレ札幌":   0.50,
    "柏レイソル":               0.48,
    "アビスパ福岡":             0.42,
    "ベガルタ仙台":             0.48,
    "ジュビロ磐田":             0.45,
    "ジェフユナイテッド千葉":   0.44,
    "大分トリニータ":           0.38,
    "V・ファーレン長崎":        0.36,
}
_DEFAULT_CORE_SUPPORT = 0.30


# ─── ダービー / 近隣対決テーブル ─────────────────────────
# ファン移動距離が近く、away でも動員が上がる組み合わせ

_DERBY_PAIRS: set[frozenset[str]] = {
    frozenset({"ガンバ大阪", "セレッソ大阪"}),            # 大阪ダービー
    frozenset({"浦和レッズ", "RB大宮アルディージャ"}),     # さいたまダービー
    frozenset({"横浜F・マリノス", "横浜FC"}),              # 横浜ダービー
    frozenset({"FC東京", "東京ヴェルディ"}),               # 東京ダービー
    frozenset({"FC東京", "FC町田ゼルビア"}),               # 多摩ダービー
    frozenset({"川崎フロンターレ", "横浜F・マリノス"}),    # 神奈川ダービー
    frozenset({"アビスパ福岡", "サガン鳥栖"}),            # 北九州ダービー
    frozenset({"鹿島アントラーズ", "浦和レッズ"}),        # 因縁マッチ
    frozenset({"ヴィッセル神戸", "ガンバ大阪"}),          # 関西ダービー
    frozenset({"ヴィッセル神戸", "セレッソ大阪"}),        # 関西ダービー
    frozenset({"名古屋グランパス", "清水エスパルス"}),    # 東海ダービー
}


def is_derby(home: str, away: str) -> bool:
    """ダービー/近隣対決かどうかを判定"""
    return frozenset({home, away}) in _DERBY_PAIRS


# ─── A. 選手移動疲労 ─────────────────────────────────────

def build_player_travel_features(
    df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    選手移動疲労の特徴量を生成。

    入力列: match_id, date, home_team, away_team, time/kickoff_time (optional)
    出力列: away_travel_distance_km, away_travel_time_proxy,
            away_long_trip_flag, days_rest_home, days_rest_away,
            night_x_long_trip, hot_humid_x_long_trip
    """
    from scripts.predict_logic import haversine_km
    from venues import get_venue_info

    if df.empty:
        return df

    if history_df is None:
        try:
            from weekend_update import load_history
            history_df = load_history()
        except Exception:
            history_df = pd.DataFrame()

    # チームごとの試合日リスト (データリーク防止)
    team_dates: dict[str, list[str]] = {}
    if not history_df.empty:
        for _, row in history_df.iterrows():
            d = str(row.get("date", ""))
            for col in ("home_team", "away_team"):
                team = str(row.get(col, ""))
                if team and d:
                    team_dates.setdefault(team, []).append(d)
        for team in team_dates:
            team_dates[team] = sorted(set(team_dates[team]))

    results = []

    for _, row in df.iterrows():
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))
        match_date = str(row.get("date", ""))

        feat: dict[str, Any] = {"match_id": row.get("match_id", "")}

        # 移動距離
        try:
            home_venue = get_venue_info(home)
            away_home_venue = get_venue_info(away)  # アウェイチームのホームスタジアム
            dist = haversine_km(
                away_home_venue["lat"], away_home_venue["lon"],
                home_venue["lat"], home_venue["lon"],
            )
            feat["away_travel_distance_km"] = dist
            feat["away_travel_time_proxy"] = round(dist / TRAVEL_SPEED_KMH, 2)
            feat["away_long_trip_flag"] = dist >= LONG_TRIP_KM
        except Exception:
            feat["away_travel_distance_km"] = None
            feat["away_travel_time_proxy"] = None
            feat["away_long_trip_flag"] = None

        # 休養日数
        feat["days_rest_home"] = _calc_days_rest(home, match_date, team_dates)
        feat["days_rest_away"] = _calc_days_rest(away, match_date, team_dates)

        # 交互作用: night × long trip
        kickoff = str(row.get("time", row.get("kickoff_time", "")))
        kh = _parse_kickoff_hour(kickoff)
        is_night = kh is not None and kh >= NIGHT_GAME_AFTER
        is_long = feat.get("away_long_trip_flag") is True
        feat["night_x_long_trip"] = 1.0 if is_night and is_long else 0.0

        # hot_humid × long trip (天候データがあれば)
        hot_humid = row.get("hot_humid_flag")
        if hot_humid is True and is_long:
            feat["hot_humid_x_long_trip"] = 1.0
        elif hot_humid is not None:
            feat["hot_humid_x_long_trip"] = 0.0
        else:
            feat["hot_humid_x_long_trip"] = None

        results.append(feat)

    return pd.DataFrame(results)


# ─── B. ファン移動しやすさ ────────────────────────────────

def build_fan_access_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    ファン移動しやすさの特徴量を生成。

    入力列: match_id, date, home_team, away_team, time/kickoff_time (optional)
    出力列: away_fan_travel_distance_km, away_fan_access_penalty,
            weekday_penalty, late_kickoff_penalty, derby_access_relief_flag
    """
    from scripts.predict_logic import haversine_km
    from venues import get_venue_info

    if df.empty:
        return df

    results = []

    for _, row in df.iterrows():
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))
        match_date = str(row.get("date", ""))

        feat: dict[str, Any] = {"match_id": row.get("match_id", "")}

        # ファン移動距離 (= 選手と同じ、アウェイファンの本拠地→試合会場)
        try:
            home_venue = get_venue_info(home)
            away_home_venue = get_venue_info(away)
            fan_dist = haversine_km(
                away_home_venue["lat"], away_home_venue["lon"],
                home_venue["lat"], home_venue["lon"],
            )
            feat["away_fan_travel_distance_km"] = fan_dist
        except Exception:
            fan_dist = None
            feat["away_fan_travel_distance_km"] = None

        # 曜日ペナルティ (平日=ファン動員減)
        weekday_pen = _compute_weekday_penalty(match_date)
        feat["weekday_penalty"] = weekday_pen

        # 遅めキックオフペナルティ (19:30+ → アウェイファン帰宅困難)
        kickoff = str(row.get("time", row.get("kickoff_time", "")))
        kh = _parse_kickoff_hour(kickoff)
        feat["late_kickoff_penalty"] = _compute_late_kickoff_penalty(kh, fan_dist)

        # ダービーアクセス緩和 (近隣対決はペナルティ軽減)
        feat["derby_access_relief_flag"] = is_derby(home, away)

        # 総合アクセスペナルティ
        feat["away_fan_access_penalty"] = _compute_fan_access_penalty(
            fan_dist, weekday_pen, feat["late_kickoff_penalty"],
            feat["derby_access_relief_flag"],
        )

        results.append(feat)

    return pd.DataFrame(results)


def _compute_weekday_penalty(date_str: str) -> float | None:
    """平日試合のファン動員ペナルティ (0.0=週末, 1.0=火水木)"""
    try:
        dt = _date.fromisoformat(date_str)
        wd = dt.weekday()  # 0=月 ... 6=日
        if wd in (5, 6):   # 土日
            return 0.0
        elif wd == 4:       # 金曜
            return 0.2
        elif wd == 0:       # 月曜
            return 0.4
        else:               # 火水木
            return 0.7
    except (ValueError, TypeError):
        return None


def _compute_late_kickoff_penalty(
    kickoff_hour: int | None,
    fan_distance_km: float | None,
) -> float | None:
    """遅めキックオフ × 長距離 → アウェイファン帰宅困難ペナルティ"""
    if kickoff_hour is None:
        return None

    base = 0.0
    if kickoff_hour >= 20:
        base = 0.6
    elif kickoff_hour >= LATE_KICKOFF_HOUR:
        base = 0.3
    elif kickoff_hour >= NIGHT_GAME_AFTER:
        base = 0.1

    # 距離で増幅 (遠いほどペナルティ大)
    if fan_distance_km is not None and fan_distance_km > 300:
        dist_factor = min(fan_distance_km / 1000, 1.0)
        base += 0.3 * dist_factor

    return round(min(base, 1.0), 3)


def _compute_fan_access_penalty(
    distance_km: float | None,
    weekday_penalty: float | None,
    late_kickoff_penalty: float | None,
    is_derby_flag: bool,
) -> float | None:
    """
    総合アクセスペナルティ (0.0=最も行きやすい, 1.0=最も行きにくい)

    = 距離ペナルティ * 0.4 + 曜日ペナルティ * 0.3 + キックオフ時刻ペナルティ * 0.3
    - ダービー緩和: -0.15

    根拠: Buraimo et al. (2010) ホーム入場者数は距離・曜日・時間帯に有意に影響
    """
    components = []

    # 距離ペナルティ (0〜1)
    if distance_km is not None:
        dist_pen = min(distance_km / 1200, 1.0)
        components.append(("distance", dist_pen, 0.4))
    else:
        return None

    # 曜日
    if weekday_penalty is not None:
        components.append(("weekday", weekday_penalty, 0.3))
    else:
        components.append(("weekday", 0.0, 0.3))

    # キックオフ
    if late_kickoff_penalty is not None:
        components.append(("kickoff", late_kickoff_penalty, 0.3))
    else:
        components.append(("kickoff", 0.0, 0.3))

    score = sum(val * weight for _, val, weight in components)

    # ダービー緩和
    if is_derby_flag:
        score = max(0.0, score - 0.15)

    return round(min(score, 1.0), 4)


# ─── C. ファン規模 proxy ─────────────────────────────────

def build_fan_scale_features(
    df: pd.DataFrame,
    attendance_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    ファン規模 proxy の特徴量を生成。

    入力列: match_id, home_team, away_team, attendance (optional)
    出力列: club_fanbase_proxy, club_core_support_proxy,
            avg_home_attendance_rolling_10, attendance_fill_rate,
            recent_attendance_momentum, vote_engagement_proxy
    """
    from venues import get_venue_info

    if df.empty:
        return df

    # 入場者数履歴からローリング平均を構築
    rolling_attendance: dict[str, list[int]] = {}
    if attendance_history is not None and not attendance_history.empty:
        for _, row in attendance_history.sort_values("date").iterrows():
            home = str(row.get("home_team", ""))
            att = row.get("attendance")
            if home and att is not None and not pd.isna(att):
                try:
                    rolling_attendance.setdefault(home, []).append(int(att))
                except (ValueError, TypeError):
                    pass

    results = []

    for _, row in df.iterrows():
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))

        feat: dict[str, Any] = {"match_id": row.get("match_id", "")}

        # 静的 fanbase proxy
        feat["club_fanbase_proxy"] = _CLUB_FANBASE_PROXY.get(home, _DEFAULT_FANBASE)
        feat["club_core_support_proxy"] = _CLUB_CORE_SUPPORT.get(home, _DEFAULT_CORE_SUPPORT)

        # ローリング10試合平均入場者数
        hist = rolling_attendance.get(home, [])
        if len(hist) >= 3:
            recent_10 = hist[-10:]
            feat["avg_home_attendance_rolling_10"] = round(sum(recent_10) / len(recent_10))

            # モメンタム: 直近5試合平均 / 前5試合平均
            if len(hist) >= 6:
                last_5 = hist[-5:]
                prev_5 = hist[-10:-5] if len(hist) >= 10 else hist[:-5]
                if prev_5:
                    feat["recent_attendance_momentum"] = round(
                        (sum(last_5) / len(last_5)) / max(sum(prev_5) / len(prev_5), 1), 3
                    )
                else:
                    feat["recent_attendance_momentum"] = None
            else:
                feat["recent_attendance_momentum"] = None
        else:
            feat["avg_home_attendance_rolling_10"] = None
            feat["recent_attendance_momentum"] = None

        # スタジアム��容率 (現在試合の入場者数があれば)
        current_att = row.get("attendance")
        venue = get_venue_info(home)
        capacity = venue.get("capacity", 0)
        if current_att is not None and not pd.isna(current_att) and capacity > 0:
            try:
                feat["attendance_fill_rate"] = round(int(current_att) / capacity, 3)
            except (ValueError, TypeError):
                feat["attendance_fill_rate"] = None
        else:
            # 履歴からの推定 fill_rate
            if feat.get("avg_home_attendance_rolling_10") and capacity > 0:
                feat["attendance_fill_rate"] = round(
                    feat["avg_home_attendance_rolling_10"] / capacity, 3
                )
            else:
                feat["attendance_fill_rate"] = None

        # vote_engagement_proxy (後段追加用、現在は nullable)
        feat["vote_engagement_proxy"] = None

        results.append(feat)

    return pd.DataFrame(results)


# ─── 統合パイプライン ─────────────────────────────────────

def build_all_fan_travel_features(
    df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
    attendance_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    全ファン/移動特徴量を一括生成して結合。

    Parameters
    ----------
    df : match_id, date, home_team, away_team 等を含む DataFrame
    history_df : 過去試合結果 (休養日数計算用)
    attendance_history : 入場者数を含む履歴 (rolling計算用)

    Returns
    -------
    全特徴量が付与された DataFrame
    """
    if df.empty:
        return df

    # A. 選手移動疲労
    player_travel = build_player_travel_features(df, history_df)

    # B. ファン移動しやすさ
    fan_access = build_fan_access_features(df)

    # C. ファン規模
    fan_scale = build_fan_scale_features(df, attendance_history)

    # match_id で結合
    result = df.copy()
    for feat_df in [player_travel, fan_access, fan_scale]:
        if not feat_df.empty and "match_id" in feat_df.columns:
            overlap = set(result.columns) & set(feat_df.columns) - {"match_id"}
            feat_clean = feat_df.drop(columns=list(overlap), errors="ignore")
            result = result.merge(feat_clean, on="match_id", how="left")

    return result


# ─── 保存・読込 ───────────────────────────────────────────

def save_fan_travel_features(df: pd.DataFrame) -> Path:
    """ファン/移動特徴量を CSV に保存 (match_id で上書き更新)"""
    if df.empty:
        return FAN_TRAVEL_CSV

    existing = load_fan_travel_features()
    if not existing.empty and "match_id" in existing.columns and "match_id" in df.columns:
        new_ids = set(df["match_id"].dropna())
        kept = existing[~existing["match_id"].isin(new_ids)]
        merged = pd.concat([kept, df], ignore_index=True)
    else:
        merged = df

    if not merged.empty and "match_id" in merged.columns:
        merged = merged.sort_values("match_id").reset_index(drop=True)

    merged.to_csv(FAN_TRAVEL_CSV, index=False, encoding="utf-8-sig")
    logger.info("ファン/移動特徴量保存: %s (%d行)", FAN_TRAVEL_CSV, len(merged))
    return FAN_TRAVEL_CSV


def load_fan_travel_features() -> pd.DataFrame:
    """保存済みファン/移動特徴量を読み込む"""
    if FAN_TRAVEL_CSV.exists():
        try:
            return pd.read_csv(FAN_TRAVEL_CSV, encoding="utf-8-sig")
        except Exception:
            try:
                return pd.read_csv(FAN_TRAVEL_CSV, encoding="utf-8")
            except Exception:
                return pd.DataFrame()
    return pd.DataFrame()


# ─── ABテスト用特徴量セット ───────────────────────────────

LEVEL_BASELINE = []

LEVEL_1_TRAVEL = [
    "away_travel_distance_km",
    "away_travel_time_proxy",
    "away_long_trip_flag",
    "days_rest_home",
    "days_rest_away",
    "night_x_long_trip",
    "hot_humid_x_long_trip",
]

LEVEL_2_FAN_ACCESS = LEVEL_1_TRAVEL + [
    "away_fan_travel_distance_km",
    "away_fan_access_penalty",
    "weekday_penalty",
    "late_kickoff_penalty",
    "derby_access_relief_flag",
]

LEVEL_3_FAN_SCALE = LEVEL_2_FAN_ACCESS + [
    "club_fanbase_proxy",
    "club_core_support_proxy",
    "avg_home_attendance_rolling_10",
    "attendance_fill_rate",
    "recent_attendance_momentum",
]

FEATURE_LEVELS = {
    "baseline": LEVEL_BASELINE,
    "travel": LEVEL_1_TRAVEL,
    "fan_access": LEVEL_2_FAN_ACCESS,
    "fan_scale": LEVEL_3_FAN_SCALE,
}


# ─── ユーティリティ ───────────────────────────────────────

def _calc_days_rest(
    team: str, match_date: str, team_dates: dict[str, list[str]],
) -> int | None:
    """データリーク防止: match_date より前の試合のみ参照"""
    dates = team_dates.get(team, [])
    if not dates or not match_date:
        return None
    try:
        target = _date.fromisoformat(match_date)
    except ValueError:
        return None
    prev = [d for d in dates if d < match_date]
    if not prev:
        return None
    try:
        last = _date.fromisoformat(prev[-1])
        return (target - last).days
    except ValueError:
        return None


def _parse_kickoff_hour(kickoff: str) -> int | None:
    if not kickoff or kickoff == "未定":
        return None
    m = re.search(r"(\d{1,2}):", str(kickoff))
    return int(m.group(1)) if m else None
