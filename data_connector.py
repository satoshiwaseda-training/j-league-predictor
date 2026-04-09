"""
data_connector.py - データ取得統合レイヤー

各データソースからの取得を統一的な FetchResult 形式で返し、
鮮度・成功/失敗・ソース情報を一元管理する。

app.py のワンボタンパイプラインはこのモジュールを通じてデータを取得する。
data_fetcher.py の既存関数はそのまま利用し、ラッパーとして機能する。
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────
# FetchResult: 統一的な取得結果型
# ────────────────────────────────────────────────────────

@dataclass
class FetchResult:
    """データ取得結果の統一型"""
    data: Any = None
    fetched_at: str = ""
    source_name: str = ""
    is_official: bool = False
    used_cache: bool = False
    success: bool = False
    coverage_note: str = ""
    error: str = ""

    def __post_init__(self):
        if not self.fetched_at:
            self.fetched_at = datetime.now().isoformat()

    @property
    def quality_badge(self) -> str:
        """UIバッジ用の品質ラベル"""
        if not self.success:
            return "取得失敗"
        if self.used_cache:
            return "キャッシュ"
        if self.is_official:
            return "公式最新"
        return "補助ソース"

    @property
    def quality_color(self) -> str:
        """UIバッジ用の色"""
        if not self.success:
            return "#dc2626"
        if self.used_cache:
            return "#ca8a04"
        if self.is_official:
            return "#16a34a"
        return "#6366f1"


# ────────────────────────────────────────────────────────
# 個別取得関数
# ────────────────────────────────────────────────────────

def fetch_fixtures(division: str = "j1") -> FetchResult:
    """試合スケジュール (直近の未来試合)"""
    try:
        from data_fetcher import get_upcoming_matches
        data = get_upcoming_matches(division)
        return FetchResult(
            data=data, success=bool(data),
            source_name="jleague.jp/match/section",
            is_official=True,
            coverage_note=f"{len(data)}試合取得" if data else "0試合",
        )
    except Exception as e:
        logger.warning("fetch_fixtures failed: %s", e)
        return FetchResult(success=False, source_name="jleague.jp",
                           error=str(e), coverage_note="取得失敗")


def fetch_results(division: str = "j1") -> FetchResult:
    """今季完了済み試合結果"""
    try:
        from data_fetcher import get_past_results
        data = get_past_results(division)
        return FetchResult(
            data=data, success=bool(data),
            source_name="jleague.jp/match/section",
            is_official=True,
            coverage_note=f"{len(data)}試合" if data else "0試合",
        )
    except Exception as e:
        logger.warning("fetch_results failed: %s", e)
        return FetchResult(success=False, source_name="jleague.jp",
                           error=str(e))


def fetch_standings(division: str = "j1") -> FetchResult:
    """順位表"""
    try:
        from data_fetcher import get_standings
        df = get_standings(division)
        ok = isinstance(df, pd.DataFrame) and not df.empty
        return FetchResult(
            data=df, success=ok,
            source_name="jleague.jp/standings",
            is_official=True,
            coverage_note=f"{len(df)}チーム" if ok else "空",
        )
    except Exception as e:
        logger.warning("fetch_standings failed: %s", e)
        return FetchResult(success=False, source_name="jleague.jp",
                           error=str(e), data=pd.DataFrame())


def fetch_xg_stats(division: str = "j1") -> FetchResult:
    """FBref xG統計 (J1のみ)"""
    try:
        from data_fetcher import get_fbref_xg_stats
        data = get_fbref_xg_stats(division)
        return FetchResult(
            data=data, success=True,
            source_name="fbref.com",
            is_official=False,
            coverage_note=f"{len(data)}チーム" if data else "J2/J3は対象外",
        )
    except Exception as e:
        logger.warning("fetch_xg_stats failed: %s", e)
        return FetchResult(success=False, source_name="fbref.com",
                           error=str(e), data={})


def fetch_discipline(division: str = "j1") -> FetchResult:
    """カード規律統計"""
    try:
        from data_fetcher import get_team_discipline_stats
        data = get_team_discipline_stats(division)
        return FetchResult(
            data=data, success=True,
            source_name="jleague.jp/stats",
            is_official=True,
            coverage_note=f"{len(data)}チーム" if data else "0チーム",
        )
    except Exception as e:
        logger.warning("fetch_discipline failed: %s", e)
        return FetchResult(success=False, source_name="jleague.jp/stats",
                           error=str(e), data={})


def fetch_team_form(team: str, n: int = 5) -> FetchResult:
    """チーム直近フォーム"""
    try:
        from data_fetcher import get_team_recent_form
        data = get_team_recent_form(team, n=n)
        return FetchResult(
            data=data, success=bool(data),
            source_name="jleague.jp",
            is_official=True,
            coverage_note=f"直近{len(data)}試合" if data else "取得失敗",
        )
    except Exception as e:
        return FetchResult(success=False, source_name="jleague.jp",
                           error=str(e), data=[])


def fetch_h2h(home: str, away: str) -> FetchResult:
    """H2H対戦成績"""
    try:
        from data_fetcher import get_head_to_head
        data = get_head_to_head(home, away)
        return FetchResult(
            data=data, success=bool(data),
            source_name="pseudo-random(seed)",
            is_official=False,
            coverage_note=f"過去{data.get('total',0)}試合",
        )
    except Exception as e:
        return FetchResult(success=False, data={}, error=str(e))


def fetch_injuries(team: str) -> FetchResult:
    """怪我・出場停止情報"""
    try:
        from data_fetcher import get_injury_news
        data = get_injury_news(team)
        return FetchResult(
            data=data, success=True,
            source_name="web-search/cache",
            is_official=False,
            coverage_note=f"{len(data)}名" if data else "情報なし",
        )
    except Exception as e:
        return FetchResult(success=False, data=[], error=str(e))


def build_elo(division: str = "j1") -> FetchResult:
    """ELOレーティング構築"""
    try:
        from scripts.predict_logic import EloSystem
        from data_fetcher import get_past_results
        past = get_past_results(division)
        elo = EloSystem(k=32.0, initial=1500.0, home_bonus=50.0)
        for r in past:
            if r.get("winner"):
                elo.update(r["home"], r["away"], r["winner"])
        return FetchResult(
            data=elo.ratings.copy(),
            success=True,
            source_name="ELO(jleague.jp結果)",
            is_official=True,
            coverage_note=f"{len(elo.ratings)}チーム, {len(past)}試合反映",
        )
    except Exception as e:
        logger.warning("build_elo failed: %s", e)
        return FetchResult(success=False, data={}, error=str(e))


# ────────────────────────────────────────────────────────
# 統合パイプライン
# ────────────────────────────────────────────────────────

@dataclass
class PipelineSnapshot:
    """ワンボタンパイプライン全体の取得結果"""
    fixtures: FetchResult = field(default_factory=FetchResult)
    results: FetchResult = field(default_factory=FetchResult)
    standings: FetchResult = field(default_factory=FetchResult)
    xg: FetchResult = field(default_factory=FetchResult)
    discipline: FetchResult = field(default_factory=FetchResult)
    elo: FetchResult = field(default_factory=FetchResult)
    generated_at: str = ""

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now().isoformat()

    @property
    def source_summary(self) -> list[dict]:
        """UIに表示するソース一覧"""
        items = []
        for name, fr in [
            ("試合日程", self.fixtures),
            ("試合結果", self.results),
            ("順位表", self.standings),
            ("xG統計", self.xg),
            ("規律統計", self.discipline),
            ("ELO", self.elo),
        ]:
            items.append({
                "name": name,
                "source": fr.source_name,
                "badge": fr.quality_badge,
                "color": fr.quality_color,
                "note": fr.coverage_note,
                "fetched_at": fr.fetched_at[:16],
            })
        return items

    @property
    def all_success(self) -> bool:
        return all(fr.success for fr in [
            self.fixtures, self.results, self.standings,
            self.xg, self.discipline, self.elo,
        ])

    @property
    def skipped_names(self) -> list[str]:
        return [name for name, fr in [
            ("試合日程", self.fixtures),
            ("試合結果", self.results),
            ("順位表", self.standings),
            ("xG統計", self.xg),
            ("規律統計", self.discipline),
            ("ELO", self.elo),
        ] if not fr.success]


def run_data_pipeline(division: str = "j1", progress_cb=None) -> PipelineSnapshot:
    """
    データ取得統合パイプライン。
    progress_cb: (step, total, label) を受け取るコールバック (Streamlit progress用)
    """
    total = 6
    snap = PipelineSnapshot()

    def _progress(step, label):
        if progress_cb:
            progress_cb(step, total, label)

    _progress(1, "試合スケジュール取得中...")
    snap.fixtures = fetch_fixtures(division)

    _progress(2, "今季試合結果取得中...")
    snap.results = fetch_results(division)

    _progress(3, "順位表取得中...")
    snap.standings = fetch_standings(division)

    _progress(4, "xG・詳細統計取得中...")
    snap.xg = fetch_xg_stats(division)

    _progress(5, "規律統計取得中...")
    snap.discipline = fetch_discipline(division)

    _progress(6, "ELOレーティング構築中...")
    snap.elo = build_elo(division)

    snap.generated_at = datetime.now().isoformat()
    return snap


# ────────────────────────────────────────────────────────
# Feature Snapshot (予測時の特徴量スナップショット)
# ────────────────────────────────────────────────────────

def build_feature_snapshot(
    match: dict,
    prediction: dict,
    contributions: dict,
    pipeline: PipelineSnapshot,
    gemini_used: bool = False,
) -> dict:
    """予測ごとの特徴量スナップショットを生成"""
    home, away = match["home"], match["away"]
    elo_ratings = pipeline.elo.data or {}
    return {
        "match_id": f"{match.get('date','?')}_{home}_{away}",
        "generated_at": datetime.now().isoformat(),
        "model_version": "v7_refined",
        "gemini_used": gemini_used,
        "source_freshness": {
            item["name"]: {"badge": item["badge"], "fetched_at": item["fetched_at"]}
            for item in pipeline.source_summary
        },
        "elo_snapshot": {
            "home": elo_ratings.get(home, 1500.0),
            "away": elo_ratings.get(away, 1500.0),
        },
        "raw_home_advantage": contributions.get("raw_home_advantage", 0),
        "closeness": contributions.get("closeness", 0.5),
        "predicted_probs": {
            "home": prediction.get("home_win_prob", 0),
            "draw": prediction.get("draw_prob", 0),
            "away": prediction.get("away_win_prob", 0),
        },
    }
