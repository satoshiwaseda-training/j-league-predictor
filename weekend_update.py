"""
weekend_update.py - 今週末結果の取得・履歴反映・特徴量更新

責務:
  1. 直近土日の終了済み試合を取得 (fetch_weekend_results)
  2. 既存履歴への安全な追記 (merge_completed_matches_into_history)
  3. 特徴量更新 (rebuild_post_result_features / update_team_state_after_results)

データリーク防止:
  - completed / 試合終了 の試合のみ取り込む
  - 未来日付の試合を結果扱いしない
  - 各試合の特徴量はその試合開始前までの情報で再計算
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timedelta, date as _date
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

RESULTS_RAW_PATH = DATA_DIR / "weekend_results_raw.csv"
RESULTS_MERGED_PATH = DATA_DIR / "weekend_results_merged.csv"
TEAM_STATE_PATH = DATA_DIR / "updated_team_state.csv"
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


# ─── チーム名正規化 ───────────────────────────────────────

def _normalize_team_name(name: str) -> str:
    """表記ゆれ吸収 (NFKC + 既知マッピング)"""
    import unicodedata
    name = unicodedata.normalize("NFKC", name).strip()
    return _TEAM_ALIAS.get(name, name)


_TEAM_ALIAS: dict[str, str] = {
    "横浜FM": "横浜F・マリノス",
    "横浜F.マリノス": "横浜F・マリノス",
    "横浜Fマリノス": "横浜F・マリノス",
    "G大阪": "ガンバ大阪",
    "Ｇ大阪": "ガンバ大阪",
    "C大阪": "セレッソ大阪",
    "Ｃ大阪": "セレッソ大阪",
    "川崎F": "川崎フロンターレ",
    "川崎Ｆ": "川崎フロンターレ",
    "東京V": "東京ヴェルディ",
    "東京Ｖ": "東京ヴェルディ",
    "RB大宮": "RB大宮アルディージャ",
}


# ─── 正規化キー生成 ─────────────────────────────────────────

def make_match_key(date_str: str, home: str, away: str, competition: str = "") -> str:
    """重複判定用の正規化キー"""
    h = _normalize_team_name(home)
    a = _normalize_team_name(away)
    return f"{date_str}_{h}_{a}_{competition}".strip("_")


def make_match_id(date_str: str, home: str, away: str, competition: str = "") -> str:
    """match_id 生成 (正規化キーベース)"""
    return make_match_key(date_str, home, away, competition)


# ─── 今週末の日付範囲 ──────────────────────────────────────

def get_weekend_range(
    reference: _date | None = None,
    extend_friday: bool = False,
    extend_monday: bool = False,
) -> tuple[_date, _date]:
    """
    直近の土日を返す。月曜早朝に実行しても直近土日を対象にする。
    reference: 基準日 (None=今日, Asia/Tokyo想定)
    extend_friday: 金曜夜開催も含めるなら True
    extend_monday: 月曜早朝開催も含めるなら True
    """
    if reference is None:
        reference = _date.today()

    weekday = reference.weekday()  # 0=月 ... 6=日

    if weekday == 5:  # 土曜
        saturday = reference
    elif weekday == 6:  # 日曜
        saturday = reference - timedelta(days=1)
    elif weekday == 0:  # 月曜 → 直前の土曜
        saturday = reference - timedelta(days=2)
    else:  # 火〜金 → 直前の土曜
        saturday = reference - timedelta(days=weekday + 2)

    sunday = saturday + timedelta(days=1)

    start = saturday - timedelta(days=1) if extend_friday else saturday
    end = sunday + timedelta(days=1) if extend_monday else sunday

    return start, end


# ─── 1. 今週末結果の自動取得 ──────────────────────────────

def fetch_weekend_results(
    divisions: list[str] | None = None,
    reference_date: _date | None = None,
    extend_friday: bool = False,
    extend_monday: bool = False,
) -> pd.DataFrame:
    """
    今週末の終了済み試合を取得して DataFrame 化。

    Parameters
    ----------
    divisions : 対象ディビジョン。None なら ["j1", "j2"]
    reference_date : 基準日
    extend_friday : 金曜開催を含めるか
    extend_monday : 月曜開催を含めるか

    Returns
    -------
    DataFrame with columns:
        match_id, competition, season, date, home_team, away_team,
        home_score, away_score, result, status, round, home_xg, away_xg, source
    """
    from data_fetcher import get_past_results

    if divisions is None:
        divisions = ["j1", "j2"]

    start, end = get_weekend_range(reference_date, extend_friday, extend_monday)
    start_str = start.isoformat()
    end_str = end.isoformat()

    logger.info("週末結果取得: %s 〜 %s, divisions=%s", start_str, end_str, divisions)

    all_rows: list[dict] = []
    errors: list[str] = []

    for div in divisions:
        try:
            past = get_past_results(div)
            for m in past:
                d = m.get("date", "")
                if d < start_str or d > end_str:
                    continue
                # 完了済みのみ (get_past_results は試合終了のみ返す設計)
                home = _normalize_team_name(m.get("home", ""))
                away = _normalize_team_name(m.get("away", ""))

                # result コード
                winner = m.get("winner", "")
                if winner == "home":
                    result_code = "H"
                elif winner == "away":
                    result_code = "A"
                elif winner == "draw":
                    result_code = "D"
                else:
                    result_code = ""

                all_rows.append({
                    "match_id": make_match_id(d, home, away, div.upper()),
                    "competition": div.upper(),
                    "season": 2026,
                    "date": d,
                    "home_team": home,
                    "away_team": away,
                    "home_score": m.get("home_score"),
                    "away_score": m.get("away_score"),
                    "result": result_code,
                    "status": "completed",
                    "round": m.get("section"),
                    "home_xg": None,
                    "away_xg": None,
                    "source": "jleague.jp",
                })
        except Exception as e:
            msg = f"{div}: 取得失敗 - {e}"
            logger.warning(msg)
            errors.append(msg)

    df = pd.DataFrame(all_rows)

    # xG 補完 (J1のみ、FBref)
    if not df.empty:
        df = _supplement_xg(df)

    # 未来日付の安全フィルタ (データリーク防止)
    today_str = _date.today().isoformat()
    if not df.empty:
        before = len(df)
        df = df[df["date"] <= today_str].copy()
        filtered = before - len(df)
        if filtered > 0:
            logger.warning("未来日付 %d 試合を除外 (データリーク防止)", filtered)

    # raw 保存
    if not df.empty:
        df.to_csv(RESULTS_RAW_PATH, index=False, encoding="utf-8-sig")
        logger.info("raw結果保存: %s (%d試合)", RESULTS_RAW_PATH, len(df))

    return df


def _supplement_xg(df: pd.DataFrame) -> pd.DataFrame:
    """FBref xG で J1 試合の xG を補完"""
    try:
        from data_fetcher import get_fbref_xg_stats
        xg_data = get_fbref_xg_stats("j1")
        if not xg_data:
            return df

        for idx, row in df.iterrows():
            if row["competition"] != "J1":
                continue
            home_xg = xg_data.get(row["home_team"], {})
            away_xg = xg_data.get(row["away_team"], {})
            if home_xg:
                df.at[idx, "home_xg"] = home_xg.get("xg_for")
            if away_xg:
                df.at[idx, "away_xg"] = away_xg.get("xg_for")
    except Exception as e:
        logger.warning("xG補完失敗: %s", e)
    return df


# ─── 2. 既存履歴への安全な追記 ────────────────────────────

HISTORY_PATH = DATA_DIR / "match_history.csv"


def load_history() -> pd.DataFrame:
    """既存の試合履歴を読み込む"""
    if HISTORY_PATH.exists():
        try:
            return pd.read_csv(HISTORY_PATH, encoding="utf-8-sig")
        except Exception:
            return pd.read_csv(HISTORY_PATH, encoding="utf-8")
    return pd.DataFrame()


def save_history(df: pd.DataFrame) -> None:
    """履歴を CSV 保存"""
    df.to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")


def merge_completed_matches_into_history(
    df_new: pd.DataFrame,
    df_hist: pd.DataFrame | None = None,
) -> dict:
    """
    新しい完了済み試合を履歴に安全に追記。

    Returns
    -------
    dict with keys:
        merged_df: マージ後 DataFrame
        new_count: 新規追加数
        updated_count: 更新数
        duplicate_count: スキップ (変更なし重複) 数
        warnings: 矛盾警告リスト
        error_count: エラー数
    """
    if df_hist is None:
        df_hist = load_history()

    result = {
        "merged_df": df_hist.copy() if not df_hist.empty else pd.DataFrame(),
        "new_count": 0,
        "updated_count": 0,
        "duplicate_count": 0,
        "warnings": [],
        "error_count": 0,
    }

    if df_new.empty:
        logger.info("新規結果なし")
        return result

    # 既存の match_id セット
    existing_ids: set[str] = set()
    existing_keys: set[str] = set()
    if not df_hist.empty:
        if "match_id" in df_hist.columns:
            existing_ids = set(df_hist["match_id"].dropna().astype(str))
        for _, row in df_hist.iterrows():
            k = make_match_key(
                str(row.get("date", "")),
                str(row.get("home_team", "")),
                str(row.get("away_team", "")),
                str(row.get("competition", "")),
            )
            existing_keys.add(k)

    new_rows: list[dict] = []
    updated_ids: list[str] = []

    for _, row in df_new.iterrows():
        mid = str(row.get("match_id", ""))
        key = make_match_key(
            str(row.get("date", "")),
            str(row.get("home_team", "")),
            str(row.get("away_team", "")),
            str(row.get("competition", "")),
        )

        # match_id で重複チェック
        if mid and mid in existing_ids:
            # 既存が completed 同士で矛盾チェック
            if not df_hist.empty and "match_id" in df_hist.columns:
                mask = df_hist["match_id"] == mid
                if mask.any():
                    existing_row = df_hist[mask].iloc[0]
                    ex_status = str(existing_row.get("status", ""))
                    new_status = str(row.get("status", ""))

                    if ex_status == "completed" and new_status == "completed":
                        ex_score = f"{existing_row.get('home_score')}-{existing_row.get('away_score')}"
                        new_score = f"{row.get('home_score')}-{row.get('away_score')}"
                        if ex_score != new_score:
                            warn = f"矛盾: {mid} 既存={ex_score} vs 新規={new_score}"
                            logger.warning(warn)
                            result["warnings"].append(warn)
                    elif ex_status != "completed" and new_status == "completed":
                        # 未終了→完了への更新は許可
                        df_hist.loc[mask, :] = row.values
                        updated_ids.append(mid)
                        result["updated_count"] += 1
                        continue

            result["duplicate_count"] += 1
            continue

        # 複合キーで重複チェック
        if key in existing_keys:
            result["duplicate_count"] += 1
            continue

        new_rows.append(row.to_dict())
        existing_ids.add(mid)
        existing_keys.add(key)
        result["new_count"] += 1

    if new_rows:
        df_append = pd.DataFrame(new_rows)
        merged = pd.concat([df_hist, df_append], ignore_index=True)
    else:
        merged = df_hist.copy()

    # 日付順ソート
    if not merged.empty and "date" in merged.columns:
        merged = merged.sort_values("date", ascending=True).reset_index(drop=True)

    result["merged_df"] = merged

    # 保存
    save_history(merged)
    merged.to_csv(RESULTS_MERGED_PATH, index=False, encoding="utf-8-sig")
    logger.info(
        "履歴マージ完了: 新規=%d, 更新=%d, 重複=%d, 警告=%d",
        result["new_count"], result["updated_count"],
        result["duplicate_count"], len(result["warnings"]),
    )

    return result


# ─── 3. 特徴量更新 ────────────────────────────────────────

def rebuild_post_result_features(
    history_df: pd.DataFrame,
    target_date: str | None = None,
) -> pd.DataFrame:
    """
    履歴データから各試合の「試合前時点」の特徴量を再計算。
    データリーク防止: 各試合の特徴量はその試合の日付より前のデータのみで計算。

    Parameters
    ----------
    history_df : 全試合履歴
    target_date : この日付以降の試合のみ再計算 (None=全件)

    Returns
    -------
    DataFrame with feature columns appended
    """
    if history_df.empty:
        return history_df

    df = history_df.sort_values("date").copy()

    # 特徴量列の初期化
    feature_cols = [
        "feat_home_elo", "feat_away_elo",
        "feat_home_form", "feat_away_form",
        "feat_home_gf_avg", "feat_away_gf_avg",
        "feat_home_ga_avg", "feat_away_ga_avg",
        "feat_home_points", "feat_away_points",
        "feat_home_wins", "feat_away_wins",
        "feat_home_draws", "feat_away_draws",
        "feat_home_losses", "feat_away_losses",
    ]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = None

    # ELO システム
    from scripts.predict_logic import EloSystem
    elo = EloSystem(k=32.0, initial=1500.0, home_bonus=50.0)

    # チーム状態追跡
    team_state: dict[str, dict] = {}

    def _get_state(team: str) -> dict:
        if team not in team_state:
            team_state[team] = {
                "form": [],  # 直近5試合 W/D/L
                "goals_for": [],
                "goals_against": [],
                "points": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "games": 0,
            }
        return team_state[team]

    for idx, row in df.iterrows():
        match_date = str(row.get("date", ""))
        if target_date and match_date < target_date:
            # 古い試合は状態更新のみ (特徴量書き込みスキップ)
            pass
        else:
            # 試合前の状態を特徴量として記録 (データリーク防止)
            home = str(row.get("home_team", ""))
            away = str(row.get("away_team", ""))
            h_state = _get_state(home)
            a_state = _get_state(away)

            df.at[idx, "feat_home_elo"] = elo.get(home)
            df.at[idx, "feat_away_elo"] = elo.get(away)

            # フォーム (直近5試合の勝率)
            h_form = h_state["form"][-5:]
            a_form = a_state["form"][-5:]
            df.at[idx, "feat_home_form"] = (
                sum(1.0 if r == "W" else 0.5 if r == "D" else 0.0 for r in h_form) / max(len(h_form), 1)
            )
            df.at[idx, "feat_away_form"] = (
                sum(1.0 if r == "W" else 0.5 if r == "D" else 0.0 for r in a_form) / max(len(a_form), 1)
            )

            # 得失点移動平均 (直近5試合)
            h_gf = h_state["goals_for"][-5:]
            h_ga = h_state["goals_against"][-5:]
            a_gf = a_state["goals_for"][-5:]
            a_ga = a_state["goals_against"][-5:]
            df.at[idx, "feat_home_gf_avg"] = sum(h_gf) / max(len(h_gf), 1)
            df.at[idx, "feat_away_gf_avg"] = sum(a_gf) / max(len(a_gf), 1)
            df.at[idx, "feat_home_ga_avg"] = sum(h_ga) / max(len(h_ga), 1)
            df.at[idx, "feat_away_ga_avg"] = sum(a_ga) / max(len(a_ga), 1)

            # 累積成績
            df.at[idx, "feat_home_points"] = h_state["points"]
            df.at[idx, "feat_away_points"] = a_state["points"]
            df.at[idx, "feat_home_wins"] = h_state["wins"]
            df.at[idx, "feat_away_wins"] = a_state["wins"]
            df.at[idx, "feat_home_draws"] = h_state["draws"]
            df.at[idx, "feat_away_draws"] = a_state["draws"]
            df.at[idx, "feat_home_losses"] = h_state["losses"]
            df.at[idx, "feat_away_losses"] = a_state["losses"]

        # --- 試合結果で状態を更新 (この試合の後) ---
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))
        result_code = str(row.get("result", ""))
        h_score = row.get("home_score")
        a_score = row.get("away_score")

        if result_code and h_score is not None and a_score is not None:
            h_state = _get_state(home)
            a_state = _get_state(away)

            # ELO 更新
            winner = "home" if result_code == "H" else ("away" if result_code == "A" else "draw")
            elo.update(home, away, winner)

            # フォーム更新
            if result_code == "H":
                h_state["form"].append("W")
                a_state["form"].append("L")
                h_state["wins"] += 1
                a_state["losses"] += 1
                h_state["points"] += 3
            elif result_code == "A":
                h_state["form"].append("L")
                a_state["form"].append("W")
                h_state["losses"] += 1
                a_state["wins"] += 1
                a_state["points"] += 3
            else:
                h_state["form"].append("D")
                a_state["form"].append("D")
                h_state["draws"] += 1
                a_state["draws"] += 1
                h_state["points"] += 1
                a_state["points"] += 1

            try:
                h_state["goals_for"].append(int(h_score))
                h_state["goals_against"].append(int(a_score))
                a_state["goals_for"].append(int(a_score))
                a_state["goals_against"].append(int(h_score))
            except (ValueError, TypeError):
                pass

            h_state["games"] += 1
            a_state["games"] += 1

    return df


def update_team_state_after_results(
    history_df: pd.DataFrame,
) -> dict[str, dict]:
    """
    全履歴から最新のチーム状態を構築。次節予測の入力用。

    Returns
    -------
    {team_name: {elo, form, gf_avg, ga_avg, points, wins, draws, losses, games}}
    """
    if history_df.empty:
        return {}

    from scripts.predict_logic import EloSystem
    elo = EloSystem(k=32.0, initial=1500.0, home_bonus=50.0)
    team_state: dict[str, dict] = {}

    def _get(team: str) -> dict:
        if team not in team_state:
            team_state[team] = {
                "elo": 1500.0,
                "form": [],
                "goals_for": [],
                "goals_against": [],
                "points": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "games": 0,
            }
        return team_state[team]

    df = history_df.sort_values("date")

    for _, row in df.iterrows():
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))
        result_code = str(row.get("result", ""))
        h_score = row.get("home_score")
        a_score = row.get("away_score")

        if not result_code or h_score is None or a_score is None:
            continue

        h_state = _get(home)
        a_state = _get(away)

        winner = "home" if result_code == "H" else ("away" if result_code == "A" else "draw")
        elo.update(home, away, winner)

        if result_code == "H":
            h_state["form"].append("W")
            a_state["form"].append("L")
            h_state["wins"] += 1
            a_state["losses"] += 1
            h_state["points"] += 3
        elif result_code == "A":
            h_state["form"].append("L")
            a_state["form"].append("W")
            h_state["losses"] += 1
            a_state["wins"] += 1
            a_state["points"] += 3
        else:
            h_state["form"].append("D")
            a_state["form"].append("D")
            h_state["draws"] += 1
            a_state["draws"] += 1
            h_state["points"] += 1
            a_state["points"] += 1

        try:
            h_state["goals_for"].append(int(h_score))
            h_state["goals_against"].append(int(a_score))
            a_state["goals_for"].append(int(a_score))
            a_state["goals_against"].append(int(h_score))
        except (ValueError, TypeError):
            pass

        h_state["games"] += 1
        a_state["games"] += 1

    # ELO を最終状態に書き込み
    for team in team_state:
        team_state[team]["elo"] = elo.get(team)
        # 直近5試合のフォーム値を算出
        recent = team_state[team]["form"][-5:]
        team_state[team]["form_score"] = (
            sum(1.0 if r == "W" else 0.5 if r == "D" else 0.0 for r in recent)
            / max(len(recent), 1)
        )
        gf = team_state[team]["goals_for"][-5:]
        ga = team_state[team]["goals_against"][-5:]
        team_state[team]["gf_avg"] = sum(gf) / max(len(gf), 1)
        team_state[team]["ga_avg"] = sum(ga) / max(len(ga), 1)
        # form は直近5だけ保持
        team_state[team]["form"] = team_state[team]["form"][-5:]
        team_state[team]["goals_for"] = gf
        team_state[team]["goals_against"] = ga

    # CSV 保存
    rows = []
    for team, st in sorted(team_state.items()):
        rows.append({
            "team": team,
            "elo": round(st["elo"], 1),
            "form_score": round(st["form_score"], 3),
            "form_last5": "".join(st["form"]),
            "gf_avg": round(st["gf_avg"], 2),
            "ga_avg": round(st["ga_avg"], 2),
            "points": st["points"],
            "wins": st["wins"],
            "draws": st["draws"],
            "losses": st["losses"],
            "games": st["games"],
        })
    if rows:
        pd.DataFrame(rows).to_csv(TEAM_STATE_PATH, index=False, encoding="utf-8-sig")
        logger.info("チーム状態保存: %s (%d チーム)", TEAM_STATE_PATH, len(rows))

    return team_state


# ─── 予測ストアへの実結果反映 ──────────────────────────────

def sync_results_to_prediction_store(df_results: pd.DataFrame) -> dict:
    """
    今週末の結果を prediction_store の予測レコードに反映。

    Returns
    -------
    {"matched": int, "updated": int, "not_found": int}
    """
    from prediction_store import load_all, update_actual

    predictions = load_all()
    stats = {"matched": 0, "updated": 0, "not_found": 0}

    if df_results.empty:
        return stats

    # 予測の _key → id マッピング
    pred_map: dict[str, dict] = {}
    for p in predictions:
        key = p.get("_key", "")
        if key:
            pred_map[key] = p

    for _, row in df_results.iterrows():
        date_str = str(row.get("date", ""))
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))
        result_code = str(row.get("result", ""))

        # prediction_store の _key 形式: "YYYY-MM-DD_HOME_AWAY"
        pkey = f"{date_str}_{home}_{away}"

        if pkey not in pred_map:
            stats["not_found"] += 1
            continue

        stats["matched"] += 1
        pred = pred_map[pkey]

        # 既に actual がある場合はスキップ
        if pred.get("actual") and pred["actual"].get("winner"):
            continue

        score = f"{row.get('home_score', '?')}-{row.get('away_score', '?')}"
        if result_code == "H":
            winner_label = "ホーム勝利"
        elif result_code == "A":
            winner_label = "アウェー勝利"
        elif result_code == "D":
            winner_label = "引き分け"
        else:
            continue

        if update_actual(pred["id"], score, winner_label):
            stats["updated"] += 1

    logger.info(
        "予測ストア反映: matched=%d, updated=%d, not_found=%d",
        stats["matched"], stats["updated"], stats["not_found"],
    )
    return stats


# ─── 統合パイプライン ─────────────────────────────────────

def run_weekend_update(
    divisions: list[str] | None = None,
    reference_date: _date | None = None,
    extend_friday: bool = False,
    extend_monday: bool = False,
) -> dict:
    """
    今週末の結果取得→履歴反映→予測ストア同期を一括実行。

    Returns
    -------
    {
        "results_df": DataFrame,
        "merge_stats": dict,
        "store_sync": dict,
        "weekend_range": (start, end),
        "errors": list[str],
    }
    """
    errors: list[str] = []

    # 1. 結果取得
    try:
        df = fetch_weekend_results(divisions, reference_date, extend_friday, extend_monday)
    except Exception as e:
        logger.error("結果取得失敗: %s", e)
        errors.append(f"結果取得失敗: {e}")
        df = pd.DataFrame()

    # 2. 履歴反映
    try:
        merge_stats = merge_completed_matches_into_history(df)
    except Exception as e:
        logger.error("履歴反映失敗: %s", e)
        errors.append(f"履歴反映失敗: {e}")
        merge_stats = {"new_count": 0, "updated_count": 0, "duplicate_count": 0, "warnings": [], "error_count": 1}

    # 3. 予測ストアへの実結果同期
    try:
        store_sync = sync_results_to_prediction_store(df)
    except Exception as e:
        logger.error("予測ストア同期失敗: %s", e)
        errors.append(f"予測ストア同期失敗: {e}")
        store_sync = {"matched": 0, "updated": 0, "not_found": 0}

    start, end = get_weekend_range(reference_date, extend_friday, extend_monday)

    # ログ保存
    _save_update_log({
        "timestamp": datetime.now().isoformat(),
        "weekend_range": [start.isoformat(), end.isoformat()],
        "divisions": divisions or ["j1", "j2"],
        "results_count": len(df),
        "merge_stats": {k: v for k, v in merge_stats.items() if k != "merged_df"},
        "store_sync": store_sync,
        "errors": errors,
    })

    return {
        "results_df": df,
        "merge_stats": merge_stats,
        "store_sync": store_sync,
        "weekend_range": (start, end),
        "errors": errors,
    }


def _save_update_log(data: dict) -> None:
    log_path = LOG_DIR / "weekend_update.log"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        logger.warning("ログ保存失敗: %s", e)
