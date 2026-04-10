"""
app.py - サトシのJリーグ勝敗予測システム
Streamlit ダッシュボード v3
"""

from __future__ import annotations

import os
import logging
import traceback
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from data_fetcher import (
    get_standings, get_upcoming_matches,
    get_team_recent_form, get_head_to_head, get_injury_news,
    get_past_results, get_fbref_xg_stats,
    get_team_discipline_stats, calc_match_interval,
)
from weather import get_weather_forecast, get_historical_weather, weather_emoji, condition_color
from scripts.predict_logic import advantage_to_probs
from venues import get_venue_info, TEAM_HOME_VENUES
from scripts.predict_logic import (
    calculate_parameter_contributions,
    predict_with_gemini,
    MODEL_WEIGHTS,
    _TEAM_CAPITAL_SCORES, _DEFAULT_CAPITAL_SCORE,
)
from prediction_store import (
    load_all as store_load_all,
    save_prediction as store_save,
    update_actual as store_update_actual,
    delete_prediction as store_delete,
    get_accuracy_stats,
)
try:
    from scripts.feedback_loop import (
        analyze_predictions,
        ask_gemini_for_analysis,
        ask_gemini_to_implement_indicators,
        sync_results_to_store,
    )
    _FEEDBACK_OK = True
except Exception as _feedback_err:
    _FEEDBACK_OK = False
    _FEEDBACK_ERR_MSG = str(_feedback_err)
    def analyze_predictions(p): return {}
    def ask_gemini_for_analysis(*a, **k): return {"error": _FEEDBACK_ERR_MSG}
    def ask_gemini_to_implement_indicators(*a, **k): return {"error": _FEEDBACK_ERR_MSG}
    def sync_results_to_store(*a, **k): return (0, 0)

load_dotenv()

# Streamlit Cloud secrets ブリッジ（main()内で呼び出す）
def _apply_secrets():
    """st.secrets → os.environ に転写（GEMINI_API_KEY / GOOGLE_API_KEY 両対応）"""
    if os.environ.get("GEMINI_API_KEY"):
        return
    for _key in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        try:
            _v = st.secrets[_key]
            if _v:
                os.environ["GEMINI_API_KEY"] = str(_v)
                return
        except Exception:
            pass

logging.basicConfig(level=logging.WARNING)

st.set_page_config(
    page_title="サトシのJリーグ予測",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",   # スマホはデフォルトで折りたたむ
)

# ─── CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
/* ── ベースレイアウト ─────────────────────────────────── */
[data-testid="stAppViewContainer"] { background: #f8fafc; }
[data-testid="stSidebar"]          { background: #f1f5f9; border-right: 1px solid #e2e8f0; }

/* ── タイトル ─────────────────────────────────────────── */
.hero-title {
  font-size: 2.4rem; font-weight: 900;
  background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  line-height: 1.2; margin: 0;
}
.hero-sub { color: #94a3b8; font-size: 0.88rem; margin-top: 0.2rem; }

/* ── 対戦ヘッダー ─────────────────────────────────────── */
.match-header {
  background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%);
  border: 1px solid #bfdbfe;
  border-radius: 16px; padding: 1.4rem 2rem;
  text-align: center; margin-bottom: 1.2rem;
}
.match-vs {
  font-size: 0.8rem; font-weight: 700; letter-spacing: 0.2em;
  color: #94a3b8; margin-bottom: 0.3rem;
}
.match-teams {
  font-size: 2rem; font-weight: 900; color: #0f172a;
  display: flex; align-items: center; justify-content: center; gap: 1rem;
}
.team-home { color: #2563eb; }
.team-away { color: #dc2626; }
.match-vs-badge {
  background: #e2e8f0; color: #475569;
  border-radius: 8px; padding: 4px 12px; font-size: 1rem;
}
.match-meta { color: #94a3b8; font-size: 0.82rem; margin-top: 0.5rem; }

/* ── 確率カード ───────────────────────────────────────── */
.prob-section {
  display: grid; grid-template-columns: 1fr 1fr 1fr;
  gap: 0.8rem; margin-bottom: 1rem;
}
.prob-card { border-radius: 14px; padding: 1.4rem 1rem; text-align: center; }
.prob-card-home { background: linear-gradient(135deg, #eff6ff, #dbeafe); border: 2px solid #3b82f6; }
.prob-card-draw { background: linear-gradient(135deg, #fffbeb, #fef3c7); border: 2px solid #f59e0b; }
.prob-card-away { background: linear-gradient(135deg, #fef2f2, #fee2e2); border: 2px solid #ef4444; }

.prob-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
              text-transform: uppercase; margin-bottom: 0.4rem; }
.prob-label-home { color: #2563eb; }
.prob-label-draw { color: #d97706; }
.prob-label-away { color: #dc2626; }

.prob-pct { font-size: 3.2rem; font-weight: 900; line-height: 1; }
.prob-pct-home { color: #1d4ed8; }
.prob-pct-draw { color: #92400e; }
.prob-pct-away { color: #991b1b; }

.prob-team { font-size: 0.82rem; color: #64748b; margin-top: 0.3rem; }

/* ── カード ───────────────────────────────────────────── */
.card {
  background: #ffffff; border-radius: 12px;
  padding: 1rem 1.3rem; margin-bottom: 0.8rem;
  border: 1px solid #e2e8f0;
  box-shadow: 0 1px 4px rgba(15,23,42,0.06);
}
.card-title {
  font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: #94a3b8; margin-bottom: 0.5rem;
}
.form-badge {
  display: inline-block; width: 24px; height: 24px; line-height: 24px;
  border-radius: 5px; text-align: center; font-weight: 800;
  font-size: 0.75rem; margin-right: 3px;
}
.form-W { background: #16a34a; color: #fff; }
.form-D { background: #ca8a04; color: #fff; }
.form-L { background: #dc2626; color: #fff; }

.stat-chip {
  display: inline-block; background: #eff6ff; color: #2563eb;
  border-radius: 6px; padding: 2px 8px; font-size: 0.78rem; margin: 2px;
}
.reasoning-box {
  background: #f5f3ff; border-left: 3px solid #7c3aed;
  border-radius: 0 10px 10px 0; padding: 0.9rem 1.1rem;
  color: #1e293b; font-size: 0.88rem; line-height: 1.85;
}

/* ── スマホ最適化 (≤768px) ────────────────────────────── */
@media (max-width: 768px) {
  /* タイトル縮小 */
  .hero-title { font-size: 1.5rem; }
  .hero-sub   { font-size: 0.78rem; }

  /* 対戦ヘッダー */
  .match-header { padding: 0.8rem 0.8rem; }
  .match-teams  { font-size: 1.15rem; gap: 0.4rem; flex-wrap: wrap; }

  /* 確率カード: 縦並びに変更 */
  .prob-section { grid-template-columns: 1fr; gap: 0.5rem; }
  .prob-pct     { font-size: 2.4rem; }

  /* 通常カード */
  .card { padding: 0.7rem 0.85rem; }

  /* フォームバッジ: 少し大きく（タップしやすく）*/
  .form-badge { width: 28px; height: 28px; line-height: 28px; font-size: 0.8rem; }

  /* stat-chip スペース調整 */
  .stat-chip { font-size: 0.75rem; padding: 3px 6px; }
}

/* ── 超スモール (≤480px) ────────────────────────────── */
@media (max-width: 480px) {
  .hero-title     { font-size: 1.2rem; }
  .prob-pct       { font-size: 2rem; }
  .match-teams    { font-size: 1rem; }
  .match-meta     { font-size: 0.72rem; }
  .prob-label     { font-size: 0.64rem; }
  .prob-team      { font-size: 0.72rem; }
}
</style>
""", unsafe_allow_html=True)


# ─── PWA マニフェスト注入 ────────────────────────────────

def _inject_pwa_meta():
    """
    PWA manifest リンクと apple-mobile-web-app メタを parent document に注入。
    Streamlit は <head> を直接編集できないため iframe 経由で JS を使用。
    manifest は data: URI として直接埋め込む（Streamlit Cloud では /app/static/ への
    リクエストが認証ページにリダイレクトされ CORS エラーになるため）。
    """
    import streamlit.components.v1 as _comp
    _comp.html("""
    <script>
    (function() {
        try {
            var doc = window.parent.document;
            // manifest link — data: URI でCORSを回避（静的ファイル参照はCloud authが遮断）
            if (!doc.querySelector('link[rel="manifest"]')) {
                var manifest = '{"name":"\\u30b5\\u30c8\\u30b7\\u306eJ\\u30ea\\u30fc\\u30b0\\u52dd\\u6557\\u4e88\\u6e2c","short_name":"J\\u30ea\\u30fc\\u30b0\\u4e88\\u6e2c","description":"Gemini AI \\u00d7 J\\u30ea\\u30fc\\u30b0\\u52dd\\u6557\\u4e88\\u6e2c","start_url":"/","display":"standalone","orientation":"portrait-primary","background_color":"#f8fafc","theme_color":"#3b82f6","lang":"ja","categories":["sports","utilities"],"icons":[{"src":"/app/static/icons/icon-192.png","sizes":"192x192","type":"image/png"},{"src":"/app/static/icons/icon-512.png","sizes":"512x512","type":"image/png"}]}';
                var link = doc.createElement('link');
                link.rel  = 'manifest';
                link.href = 'data:application/json,' + encodeURIComponent(manifest);
                doc.head.appendChild(link);
            }
            // Apple ホーム画面対応
            ['apple-mobile-web-app-capable', 'mobile-web-app-capable'].forEach(function(n) {
                if (!doc.querySelector('meta[name="' + n + '"]')) {
                    var m = doc.createElement('meta');
                    m.name = n; m.content = 'yes';
                    doc.head.appendChild(m);
                }
            });
            // Apple ステータスバー
            if (!doc.querySelector('meta[name="apple-mobile-web-app-status-bar-style"]')) {
                var sb = doc.createElement('meta');
                sb.name = 'apple-mobile-web-app-status-bar-style';
                sb.content = 'default';
                doc.head.appendChild(sb);
            }
            // Apple タイトル
            if (!doc.querySelector('meta[name="apple-mobile-web-app-title"]')) {
                var at = doc.createElement('meta');
                at.name = 'apple-mobile-web-app-title';
                at.content = 'J\u30ea\u30fc\u30b0\u4e88\u6e2c';
                doc.head.appendChild(at);
            }
        } catch(e) {}
    })();
    </script>
    """, height=0, scrolling=False)


# ─── キャッシュ ───────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def cached_standings(division: str) -> pd.DataFrame:
    return get_standings(division)

@st.cache_data(ttl=1800, show_spinner=False)
def cached_matches(division: str) -> list[dict]:
    return get_upcoming_matches(division)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_weather(lat: float, lon: float, match_date: str) -> dict:
    return get_weather_forecast(lat, lon, match_date)

@st.cache_data(ttl=86400, show_spinner=False)
def cached_h2h(home: str, away: str) -> dict:
    return get_head_to_head(home, away)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_form(team: str) -> list[str]:
    return get_team_recent_form(team, n=5)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_fbref_xg(division: str) -> dict:
    """FBref xGデータ (J1のみ有効、他は空辞書)"""
    return get_fbref_xg_stats(division)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_discipline(division: str) -> dict:
    """チームカード規律統計 (失敗時は空辞書 → 中立扱い)"""
    return get_team_discipline_stats(division)

@st.cache_data(ttl=1800, show_spinner=False)
def cached_past_results(division: str) -> list[dict]:
    """今季完了試合一覧 (試合間隔計算に使用)"""
    return get_past_results(division)

@st.cache_data(ttl=1800, show_spinner=False)
def cached_elo_ratings(division: str) -> dict[str, float]:
    """ELOレーティングをdictとして返す (cache_data互換)"""
    from scripts.predict_logic import EloSystem
    past = cached_past_results(division)
    elo = EloSystem(k=32.0, initial=1500.0, home_bonus=50.0)
    for r in past:
        if r.get("winner"):
            elo.update(r["home"], r["away"], r["winner"])
    return elo.ratings.copy()

def get_elo_scores(division: str, home: str, away: str) -> tuple[float, float]:
    """ELO期待勝率を取得 (キャッシュ経由)"""
    ratings = cached_elo_ratings(division)
    initial, home_bonus = 1500.0, 50.0
    rh = ratings.get(home, initial) + home_bonus
    ra = ratings.get(away, initial)
    eh = 1.0 / (1.0 + 10.0 ** ((ra - rh) / 400.0))
    return eh, 1.0 - eh


# ─── UI ヘルパー ─────────────────────────────────────────

def form_html(form: list[str]) -> str:
    return "".join(f'<span class="form-badge form-{r}">{r}</span>' for r in form)


def make_prob_gauge(h: int, d: int, a: int, home: str, away: str) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=[h, d, a],
        y=[f"🏠 {home}", "🤝 引き分け", f"✈️ {away}"],
        orientation="h",
        marker_color=["#3b82f6", "#f59e0b", "#ef4444"],
        text=[f"{h}%", f"{d}%", f"{a}%"],
        textposition="inside",
        textfont=dict(size=18, color="white", family="Arial Black"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 100], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(tickfont=dict(color="#475569", size=13)),
        margin=dict(l=0, r=0, t=5, b=5), height=160, showlegend=False,
    )
    return fig


def make_radar_chart(contributions: dict, home: str, away: str) -> go.Figure:
    labels = {
        "team_strength":   "チーム強度",
        "attack_rate":     "攻撃率",
        "defense_rate":    "守備率",
        "recent_form":     "フォーム",
        "xg_differential": "xG差分",
        "home_advantage":  "ホームADV",
        "head_to_head":    "H2H",
        "injury_impact":   "選手状態",
    }
    # contributionsに存在するキーのみ使用
    valid_keys = [k for k in labels if k in contributions]
    valid_labels = {k: labels[k] for k in valid_keys}
    cats = list(valid_labels.values())
    h_vals = [round(contributions[k]["home_score"] * 100) for k in valid_keys]
    a_vals = [round(contributions[k]["away_score"] * 100) for k in valid_keys]
    cats  += [cats[0]];  h_vals += [h_vals[0]];  a_vals += [a_vals[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=h_vals, theta=cats, fill="toself",
        name=f"🏠 {home}", line=dict(color="#3b82f6", width=2),
        fillcolor="rgba(59,130,246,0.2)"))
    fig.add_trace(go.Scatterpolar(r=a_vals, theta=cats, fill="toself",
        name=f"✈️ {away}", line=dict(color="#ef4444", width=2),
        fillcolor="rgba(239,68,68,0.2)"))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(248,250,252,1)",
            radialaxis=dict(visible=True, range=[0, 100],
                gridcolor="#e2e8f0", tickfont=dict(color="#94a3b8", size=9)),
            angularaxis=dict(tickfont=dict(color="#475569", size=11)),
        ),
        legend=dict(font=dict(color="#475569"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=30, r=30, t=30, b=30), height=370,
    )
    return fig


def make_contribution_bar(contributions: dict, home: str, away: str) -> go.Figure:
    labels = {
        "team_strength":   "チーム強度(勝点)",
        "attack_rate":     "攻撃率",
        "defense_rate":    "守備率",
        "recent_form":     "直近フォーム",
        "xg_differential": "xG差分",
        "home_advantage":  "ホームADV",
        "head_to_head":    "H2H",
        "injury_impact":   "選手状態",
        "weather_fatigue": "天気/疲労",
        "travel_distance": "移動距離",
    }
    keys = [k for k in labels.keys() if k in contributions]
    contribs = [contributions[k]["contribution"] for k in keys]
    colors = ["#3b82f6" if c >= 0 else "#ef4444" for c in contribs]

    fig = go.Figure(go.Bar(
        x=contribs, y=[labels[k] for k in keys], orientation="h",
        marker_color=colors,
        text=[f"{'+' if c>=0 else ''}{c:.4f}" for c in contribs],
        textposition="outside",
        textfont=dict(size=11, color="#94a3b8"),
    ))
    fig.add_vline(x=0, line_color="#4b5563", line_width=1)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=f"<-- {away} 有利  |  {home} 有利 -->",
            title_font=dict(color="#64748b", size=11),
            gridcolor="#e2e8f0", zerolinecolor="#cbd5e1",
            tickfont=dict(color="#64748b"),
        ),
        yaxis=dict(tickfont=dict(color="#475569", size=12)),
        margin=dict(l=10, r=70, t=15, b=50), height=300, showlegend=False,
    )
    return fig


def make_3d_scatter(standings_df: pd.DataFrame, home: str, away: str) -> go.Figure:
    if standings_df.empty:
        return go.Figure()

    df = standings_df.copy()
    df["_pts"]  = pd.to_numeric(df.get("勝点", pd.Series([0]*len(df))), errors="coerce").fillna(0)
    df["_rate"] = pd.to_numeric(df.get("勝率", pd.Series([0.5]*len(df))), errors="coerce").fillna(0.5)

    # 得失点差: "+26" → 26
    gd_col = df.get("得失点差", pd.Series(["0"]*len(df)))
    if not isinstance(gd_col, pd.Series):
        gd_col = pd.Series(["0"] * len(df))
    df["_gd"] = pd.to_numeric(
        gd_col.astype(str).str.replace("+", "", regex=False),
        errors="coerce"
    ).fillna(0)

    teams = df.get("チーム", pd.Series([""]*len(df)))
    if not isinstance(teams, pd.Series):
        teams = pd.Series([""]*len(df))

    colors, sizes = [], []
    for t in teams.tolist():
        if str(t) == home:   colors.append("#3b82f6"); sizes.append(14)
        elif str(t) == away: colors.append("#ef4444"); sizes.append(14)
        else:                colors.append("#4b5563"); sizes.append(7)

    fig = go.Figure(go.Scatter3d(
        x=df["_pts"].tolist(), y=df["_rate"].tolist(), z=df["_gd"].tolist(),
        mode="markers+text",
        marker=dict(size=sizes, color=colors, opacity=0.85,
                    line=dict(color="#1e2d3d", width=0.5)),
        text=teams.tolist(),
        textposition="top center",
        textfont=dict(size=9, color="#94a3b8"),
        hovertemplate="<b>%{text}</b><br>勝点: %{x}<br>勝率: %{y:.3f}<br>得失点差: %{z}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="rgba(248,250,252,1)",
            xaxis=dict(title="勝点",    gridcolor="#e2e8f0", color="#64748b"),
            yaxis=dict(title="勝率",    gridcolor="#e2e8f0", color="#64748b"),
            zaxis=dict(title="得失点差", gridcolor="#e2e8f0", color="#64748b"),
            # モバイルでのタッチ操作最適化
            dragmode="turntable",
        ),
        # タッチ操作: turntable はスワイプで回転（ピンチズームより直感的）
        dragmode="turntable",
        margin=dict(l=0, r=0, t=0, b=0), height=400,
    )
    return fig


# ─── サイドバー ───────────────────────────────────────────

def sidebar() -> tuple[str, dict | None]:
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:0.5rem 0 1rem;'>
          <div style='font-size:2rem;'>⚽</div>
          <div style='font-weight:900;font-size:1.05rem;color:#f1f5f9;'>サトシのJリーグ予測</div>
          <div style='font-size:0.7rem;color:#4b5563;'>Gemini 2.5 Flash 搭載</div>
        </div>""", unsafe_allow_html=True)

        division = st.selectbox(
            "リーグ",
            ["j1", "j2", "j3"],
            format_func=lambda x: {"j1": "⚽ J1リーグ", "j2": "🥈 J2リーグ", "j3": "🥉 J3リーグ"}[x],
        )

        st.markdown("---")
        st.markdown("### 対戦カード設定")
        mode = st.radio("入力方法", ["スケジュールから", "チームを手動選択"], horizontal=True, label_visibility="collapsed")

        selected: dict | None = None

        if mode == "スケジュールから":
            with st.spinner("スケジュール取得中..."):
                matches = cached_matches(division)

            if not matches:
                st.warning("スケジュール取得失敗\n手動選択に切り替えてください")
            else:
                opts = [
                    f"{m['date']}  {m['home']} vs {m['away']}"
                    for m in matches
                ]
                idx = st.selectbox("試合を選択", range(len(opts)), format_func=lambda i: opts[i])
                selected = matches[idx]

                # 選択中の試合を大きく表示
                m = matches[idx]
                st.markdown(f"""
                <div style='background:#1e2d3d;border-radius:10px;padding:0.8rem;margin-top:0.5rem;'>
                  <div style='color:#64748b;font-size:0.7rem;'>選択中の対戦</div>
                  <div style='color:#60a5fa;font-weight:700;'>{m['home']}</div>
                  <div style='color:#64748b;font-size:0.75rem;text-align:center;'>VS</div>
                  <div style='color:#f87171;font-weight:700;'>{m['away']}</div>
                  <div style='color:#64748b;font-size:0.72rem;margin-top:0.3rem;'>
                    {m.get('date','')} {m.get('time','')}
                  </div>
                </div>""", unsafe_allow_html=True)
        else:
            teams = sorted(TEAM_HOME_VENUES.keys())
            home = st.selectbox("🏠 ホームチーム", teams)
            away = st.selectbox("✈️ アウェーチーム", [t for t in teams if t != home])
            md   = st.date_input("試合日", value=date.today() + timedelta(days=3))
            vi   = get_venue_info(home)
            selected = {
                "home": home, "away": away,
                "date": md.isoformat(), "time": "19:00", "venue": vi["name"],
            }

        st.markdown("---")

        if selected:
            if st.button("🔮  予測を実行する", use_container_width=True, type="primary"):
                st.session_state["match"] = selected
                st.session_state["run"] = True
            elif "match" not in st.session_state:
                st.session_state["match"] = selected

        if st.button("🔄 データ更新", use_container_width=True):
            st.cache_data.clear()
            # 全試合予測キャッシュ（session_state）も全クリア
            for k in [k for k in st.session_state if k.startswith("all_preds_")]:
                del st.session_state[k]
            st.success("更新しました")
            st.rerun()

        st.markdown("---")
        ak = os.getenv("GEMINI_API_KEY", "")
        if not ak or ak == "your_gemini_api_key_here":
            _apply_secrets()
            ak = os.getenv("GEMINI_API_KEY", "")
        if ak and ak != "your_gemini_api_key_here":
            st.success("Gemini 2.5 Flash 接続済み ✓")
        else:
            st.error("Gemini API キー未設定\n`.env` を確認してください")
        if not _FEEDBACK_OK:
            st.warning(f"feedback_loop import error:\n{_FEEDBACK_ERR_MSG}")

    return division, st.session_state.get("match")


# ─── 予測タブ ─────────────────────────────────────────────

def render_prediction(match: dict, division: str):
    home = match.get("home", "?")
    away = match.get("away", "?")
    match_date = match.get("date", date.today().isoformat())
    venue_name = match.get("venue") or get_venue_info(home)["name"]

    # ══ 対戦ヘッダー（最重要：誰が対戦するか）══
    st.markdown(f"""
    <div class="match-header">
      <div class="match-vs">対戦カード</div>
      <div class="match-teams">
        <span class="team-home">🏠 {home}</span>
        <span class="match-vs-badge">VS</span>
        <span class="team-away">{away} ✈️</span>
      </div>
      <div class="match-meta">
        📅 {match_date}&nbsp;&nbsp;🕖 {match.get('time','')}&nbsp;&nbsp;🏟️ {venue_name}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── データ収集 ────────────────────────────────────────
    try:
        with st.spinner("データを収集・分析中..."):
            standings   = cached_standings(division)

            def row(team: str) -> dict:
                if standings.empty: return {}
                r = standings[standings["チーム"] == team]
                return r.iloc[0].to_dict() if not r.empty else {}

            home_stats  = row(home)
            away_stats  = row(away)
            home_form   = cached_form(home)
            away_form   = cached_form(away)
            h2h         = cached_h2h(home, away)
            home_inj    = get_injury_news(home)
            away_inj    = get_injury_news(away)
            home_venue  = get_venue_info(home, venue_name)
            away_venue  = get_venue_info(away)
            weather     = cached_weather(home_venue["lat"], home_venue["lon"], match_date)
            xg_data     = cached_fbref_xg(division)
            home_xg     = xg_data.get(home, {})
            away_xg     = xg_data.get(away, {})
            cards_data  = cached_discipline(division)
            home_cards  = cards_data.get(home, {})
            away_cards  = cards_data.get(away, {})
            past        = cached_past_results(division)
            home_days   = calc_match_interval(home, str(match_date), past)
            away_days   = calc_match_interval(away, str(match_date), past)
            elo_h, elo_a = get_elo_scores(division, home, away)

            contributions = calculate_parameter_contributions(
                home, away,
                home_stats, away_stats, home_form, away_form,
                h2h, weather, home_inj, away_inj, home_venue, away_venue,
                home_xg=home_xg, away_xg=away_xg,
                home_cards=home_cards, away_cards=away_cards,
                home_days=home_days, away_days=away_days,
                elo_home_score=elo_h, elo_away_score=elo_a,
            )

        with st.spinner("Gemini 2.5 Flash で予測中..."):
            prediction = predict_with_gemini(
                home, away, contributions,
                home_stats, away_stats,
                home_form, away_form, h2h, weather,
                home_xg=home_xg, away_xg=away_xg,
                home_days=home_days, away_days=away_days,
                home_cards=home_cards, away_cards=away_cards,
                home_injuries=home_inj, away_injuries=away_inj,
            )

    except Exception as exc:
        st.error(f"データ取得・分析中にエラーが発生しました: {exc}")
        st.code(traceback.format_exc())
        return

    # ── 結果を取り出す ──────────────────────────────────
    h_pct   = int(prediction.get("home_win_prob", 40))
    d_pct   = int(prediction.get("draw_prob",     25))
    a_pct   = int(prediction.get("away_win_prob", 35))
    score   = prediction.get("predicted_score", "?-?")
    conf    = prediction.get("confidence", "medium")
    upset   = int(prediction.get("upset_risk", 30))
    model   = prediction.get("model", "?")
    dist_km = prediction.get("distance_km", contributions.get("distance_km", 0))

    conf_label = {"high": "高 🟢", "medium": "中 🟡", "low": "低 🔴"}.get(conf, conf)

    # ══ 勝敗確率（最大フォント・3カラム）══
    st.markdown(f"""
    <div class="prob-section">

      <div class="prob-card prob-card-home">
        <div class="prob-label prob-label-home">🏠 ホーム勝利</div>
        <div class="prob-pct prob-pct-home">{h_pct}<span style='font-size:1.4rem;'>%</span></div>
        <div class="prob-team">{home}</div>
      </div>

      <div class="prob-card prob-card-draw">
        <div class="prob-label prob-label-draw">🤝 引き分け</div>
        <div class="prob-pct prob-pct-draw">{d_pct}<span style='font-size:1.4rem;'>%</span></div>
        <div class="prob-team">&nbsp;</div>
      </div>

      <div class="prob-card prob-card-away">
        <div class="prob-label prob-label-away">✈️ アウェー勝利</div>
        <div class="prob-pct prob-pct-away">{a_pct}<span style='font-size:1.4rem;'>%</span></div>
        <div class="prob-team">{away}</div>
      </div>

    </div>
    """, unsafe_allow_html=True)

    # ── 予想スコアと補足 ────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("⚽ 予想スコア", score)
    m2.metric("🎯 信頼度", conf_label)
    m3.metric("⚠️ 番狂わせリスク", f"{upset}%")
    m4.metric("✈️ 移動距離", f"{dist_km} km")

    # ══ v3 新指標インジケーター ══════════════════════════
    _params = contributions.get("parameters", {})
    _h_cap  = contributions.get("capital_home", 0.5)
    _a_cap  = contributions.get("capital_away", 0.5)
    _cap_diff = _h_cap - _a_cap
    _h_disc = _params.get("discipline_risk", {}).get("home_score", 0.6)
    _a_disc = _params.get("discipline_risk", {}).get("away_score", 0.6)
    _h_att  = _params.get("attrition_rate", {}).get("home_score", 0.8)
    _a_att  = _params.get("attrition_rate", {}).get("away_score", 0.8)
    _h_days = contributions.get("home_days", 0)
    _a_days = contributions.get("away_days", 0)
    _gk_prob = prediction.get("giant_killing_prob")

    def _bar(val: float, color: str) -> str:
        pct = int(val * 100)
        return (f'<div style="background:#e2e8f0;border-radius:4px;height:6px;width:100%;margin-top:2px;">'
                f'<div style="width:{pct}%;background:{color};height:6px;border-radius:4px;"></div></div>')

    def _disc_label(s: float) -> str:
        return "良好 🟢" if s >= 0.70 else ("注意 🟡" if s >= 0.50 else "危険 🔴")

    def _att_label(s: float) -> str:
        return "万全 🟢" if s >= 0.75 else ("軽微 🟡" if s >= 0.50 else "深刻 🔴")

    def _interval_label(d: int) -> str:
        if d <= 0: return "不明"
        elif d <= 2: return f"中{d}日 🔴"
        elif d == 3: return f"中{d}日 🟡"
        elif d <= 7: return f"中{d}日 🟢"
        else: return f"中{d}日 🟡"

    # ジャイアントキリング警告
    if abs(_cap_diff) >= 0.25:
        _richer = home if _cap_diff > 0 else away
        _poorer = away if _cap_diff > 0 else home
        _gk_text = ""
        if _gk_prob is not None:
            _gk_text = f"　　格下勝利確率: **{_gk_prob}%**"
        st.warning(
            f"⚡ **資本格差試合**: {_richer}（資本力{max(_h_cap,_a_cap):.0%}）vs "
            f"{_poorer}（資本力{min(_h_cap,_a_cap):.0%}）{_gk_text}",
            icon=None,
        )

    with st.expander("📊 v3 新指標インジケーター（資本力・規律・損耗率・試合間隔）", expanded=True):
        ind1, ind2, ind3, ind4 = st.columns(4)

        with ind1:
            st.markdown(f"""<div style='text-align:center;'>
              <div style='font-size:0.7rem;color:#64748b;font-weight:600;'>💰 資本力スコア</div>
              <div style='font-size:0.9rem;font-weight:700;color:#1e293b;margin:4px 0;'>{home}</div>
              <div style='font-size:1.2rem;font-weight:800;color:#2563eb;'>{_h_cap:.0%}</div>
              {_bar(_h_cap, '#2563eb')}
              <div style='font-size:0.9rem;font-weight:700;color:#1e293b;margin:8px 0 4px;'>{away}</div>
              <div style='font-size:1.2rem;font-weight:800;color:#dc2626;'>{_a_cap:.0%}</div>
              {_bar(_a_cap, '#dc2626')}
              <div style='font-size:0.65rem;color:#94a3b8;margin-top:4px;'>親会社収益・年俸総額</div>
            </div>""", unsafe_allow_html=True)

        with ind2:
            st.markdown(f"""<div style='text-align:center;'>
              <div style='font-size:0.7rem;color:#64748b;font-weight:600;'>🟡 規律リスク</div>
              <div style='font-size:0.9rem;font-weight:700;color:#1e293b;margin:4px 0;'>{home}</div>
              <div style='font-size:1rem;font-weight:700;'>{_disc_label(_h_disc)}</div>
              {_bar(_h_disc, '#f59e0b')}
              <div style='font-size:0.9rem;font-weight:700;color:#1e293b;margin:8px 0 4px;'>{away}</div>
              <div style='font-size:1rem;font-weight:700;'>{_disc_label(_a_disc)}</div>
              {_bar(_a_disc, '#f59e0b')}
              <div style='font-size:0.65rem;color:#94a3b8;margin-top:4px;'>イエロー累積・退場リスク</div>
            </div>""", unsafe_allow_html=True)

        with ind3:
            st.markdown(f"""<div style='text-align:center;'>
              <div style='font-size:0.7rem;color:#64748b;font-weight:600;'>🩹 損耗率</div>
              <div style='font-size:0.9rem;font-weight:700;color:#1e293b;margin:4px 0;'>{home}</div>
              <div style='font-size:1rem;font-weight:700;'>{_att_label(_h_att)}</div>
              {_bar(_h_att, '#10b981')}
              <div style='font-size:0.9rem;font-weight:700;color:#1e293b;margin:8px 0 4px;'>{away}</div>
              <div style='font-size:1rem;font-weight:700;'>{_att_label(_a_att)}</div>
              {_bar(_a_att, '#10b981')}
              <div style='font-size:0.65rem;color:#94a3b8;margin-top:4px;'>怪我人/スカッド比率</div>
            </div>""", unsafe_allow_html=True)

        with ind4:
            st.markdown(f"""<div style='text-align:center;'>
              <div style='font-size:0.7rem;color:#64748b;font-weight:600;'>⏱️ 試合間隔</div>
              <div style='font-size:0.9rem;font-weight:700;color:#1e293b;margin:4px 0;'>{home}</div>
              <div style='font-size:1rem;font-weight:700;'>{_interval_label(_h_days)}</div>
              <div style='font-size:0.9rem;font-weight:700;color:#1e293b;margin:8px 0 4px;'>{away}</div>
              <div style='font-size:1rem;font-weight:700;'>{_interval_label(_a_days)}</div>
              <div style='font-size:0.65rem;color:#94a3b8;margin-top:4px;'>中何日 (6-7日=最適)</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── チームデータ横並び ──────────────────────────────
    cc1, cc2 = st.columns(2)
    def _record_str(stats: dict) -> str:
        """勝敗記録を 2026 PK形式で文字列化"""
        w  = stats.get('勝', '?')
        pk = stats.get('PK勝', '')
        kl = stats.get('PK負', '')
        l  = stats.get('負', '?')
        pk_part = f" PK{pk}勝{kl}負" if pk not in ('', '?', 0) or kl not in ('', '?', 0) else ""
        return f"{w}勝{l}敗{pk_part}"

    with cc1:
        st.markdown(f"""
        <div class="card">
          <div class="card-title">🏠 {home} — データ</div>
          <div style='margin:0.4rem 0;'>
            <span class="stat-chip">{home_stats.get('順位','?')}位</span>
            <span class="stat-chip">勝点 {home_stats.get('勝点','?')}</span>
            <span class="stat-chip">{_record_str(home_stats)}</span>
          </div>
          <div style='font-size:0.75rem;color:#64748b;'>得点 {home_stats.get('得点','?')} / 失点 {home_stats.get('失点','?')} / 得失点差 {home_stats.get('得失点差','?')}</div>
          <div style='margin-top:0.5rem;font-size:0.75rem;color:#64748b;'>直近5試合</div>
          <div style='margin-top:3px;'>{form_html(home_form)}</div>
        </div>""", unsafe_allow_html=True)

    with cc2:
        st.markdown(f"""
        <div class="card">
          <div class="card-title">✈️ {away} — データ</div>
          <div style='margin:0.4rem 0;'>
            <span class="stat-chip">{away_stats.get('順位','?')}位</span>
            <span class="stat-chip">勝点 {away_stats.get('勝点','?')}</span>
            <span class="stat-chip">{_record_str(away_stats)}</span>
          </div>
          <div style='font-size:0.75rem;color:#64748b;'>得点 {away_stats.get('得点','?')} / 失点 {away_stats.get('失点','?')} / 得失点差 {away_stats.get('得失点差','?')}</div>
          <div style='margin-top:0.5rem;font-size:0.75rem;color:#64748b;'>直近5試合</div>
          <div style='margin-top:3px;'>{form_html(away_form)}</div>
        </div>""", unsafe_allow_html=True)

    # ── 天気 + H2H ────────────────────────────────────────
    wc, hc = st.columns(2)

    with wc:
        wcode   = weather.get("weather_code", 0)
        fat_pct = int(weather.get("fatigue_factor", 0.1) * 100)
        ccolor  = condition_color(weather.get("condition", "good"))
        st.markdown(f"""
        <div class="card">
          <div class="card-title">🌤️ 試合当日の天気 — {home_venue.get('city','?')}</div>
          <div style='display:flex;gap:1.2rem;margin:0.4rem 0;'>
            <div style='text-align:center;'>
              <div style='font-size:1.8rem;'>{weather_emoji(wcode)}</div>
              <div style='font-size:0.8rem;color:#1e293b;'>{weather.get('description','?')}</div>
            </div>
            <div style='text-align:center;'>
              <div style='font-size:1.3rem;font-weight:700;color:#0f172a;'>{weather.get('temp_avg','?')}°C</div>
              <div style='font-size:0.7rem;color:#64748b;'>平均気温</div>
            </div>
            <div style='text-align:center;'>
              <div style='font-size:1.3rem;font-weight:700;color:#0f172a;'>{weather.get('precipitation',0)}mm</div>
              <div style='font-size:0.7rem;color:#64748b;'>降水量</div>
            </div>
            <div style='text-align:center;'>
              <div style='font-size:1.3rem;font-weight:700;color:#0f172a;'>{weather.get('wind_speed','?')}km/h</div>
              <div style='font-size:0.7rem;color:#64748b;'>風速</div>
            </div>
          </div>
          <div style='font-size:0.72rem;color:#64748b;margin-top:0.4rem;'>移動距離による疲労（アウェー）</div>
          <div style='background:#e2e8f0;border-radius:999px;height:10px;margin-top:3px;overflow:hidden;'>
            <div style='width:{fat_pct}%;background:{ccolor};height:100%;border-radius:999px;'></div>
          </div>
        </div>""", unsafe_allow_html=True)

    with hc:
        hw = h2h.get("home_wins", 0)
        dd = h2h.get("draws", 0)
        aw = h2h.get("away_wins", 0)
        total_h2h = h2h.get("total", 0)
        recent_sc = "  ".join(r["score"] for r in h2h.get("recent", [])[:5])
        st.markdown(f"""
        <div class="card">
          <div class="card-title">📋 H2H 対戦成績（過去{total_h2h}試合）</div>
          <div style='display:flex;gap:1.5rem;justify-content:space-around;padding:0.5rem 0;'>
            <div style='text-align:center;'>
              <div style='font-size:2rem;font-weight:900;color:#2563eb;'>{hw}</div>
              <div style='font-size:0.7rem;color:#64748b;'>{home}</div>
            </div>
            <div style='text-align:center;'>
              <div style='font-size:2rem;font-weight:900;color:#d97706;'>{dd}</div>
              <div style='font-size:0.7rem;color:#64748b;'>引き分け</div>
            </div>
            <div style='text-align:center;'>
              <div style='font-size:2rem;font-weight:900;color:#dc2626;'>{aw}</div>
              <div style='font-size:0.7rem;color:#64748b;'>{away}</div>
            </div>
          </div>
          <div style='font-size:0.75rem;color:#64748b;'>
            直近: <span style='color:#94a3b8;'>{recent_sc}</span>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── AI 根拠 ───────────────────────────────────────────
    reasoning   = prediction.get("reasoning", "")
    key_factors = prediction.get("key_factors", [])
    model_notes = prediction.get("model_notes", "")

    chips = "".join(
        f'<span style="display:inline-block;background:#eff6ff;color:#2563eb;'
        f'border-radius:999px;padding:2px 10px;font-size:0.76rem;margin:2px;'
        f'border:1px solid #bfdbfe;">'
        f'▶ {f}</span>'
        for f in key_factors
    )

    st.markdown(f"""
    <div class="card" style="margin-top:0.2rem;">
      <div class="card-title">🤖 AI 科学的分析根拠 — {model}</div>
      <div style="margin-bottom:0.5rem;">{chips}</div>
      <div class="reasoning-box">{reasoning.replace(chr(10), "<br>")}</div>
      {f'<div style="margin-top:0.5rem;color:#64748b;font-size:0.76rem;">{model_notes}</div>' if model_notes else ""}
    </div>""", unsafe_allow_html=True)

    # ── ビジュアル分析タブ ────────────────────────────────
    st.markdown("---")
    st.markdown("#### 詳細パラメータ分析")
    contrib_params = contributions.get("parameters", {})

    viz1, viz2, viz3 = st.tabs(["🕸️ レーダーチャート", "📊 パラメータ貢献度", "🌐 3D チーム分布"])

    with viz1:
        st.plotly_chart(
            make_radar_chart(contrib_params, home, away),
            width="stretch", config={"displayModeBar": False},
        )
        st.caption("各パラメータのホーム・アウェースコア (0〜100)")

    with viz2:
        st.plotly_chart(
            make_contribution_bar(contrib_params, home, away),
            width="stretch", config={"displayModeBar": False},
        )
        raw = contributions.get("raw_home_advantage", 0)
        st.caption(f"加重ホームアドバンテージスコア: **{raw:+.4f}**  （正 = {home}有利）")

        with st.expander("重み付け詳細（v2 研究ベースモデル）"):
            names_jp = {
                "team_strength":    "チーム強度(勝点)",
                "attack_rate":      "攻撃率(得点/試合)",
                "defense_rate":     "守備率(失点/試合)",
                "recent_form":      "直近フォーム",
                "xg_differential":  "xG差分(FBref)",
                "home_advantage":   "ホームADV",
                "head_to_head":     "H2H",
                "injury_impact":    "選手状態",
                "weather_fatigue":  "天気/疲労",
                "travel_distance":  "移動距離",
            }
            rows = []
            for k, v in contrib_params.items():
                rows.append({
                    "パラメータ": names_jp.get(k, k),
                    "重み": f"{MODEL_WEIGHTS.get(k,0):.0%}",
                    f"{home}スコア": f"{v['home_score']:.3f}",
                    f"{away}スコア": f"{v['away_score']:.3f}",
                    "差": f"{v['home_advantage']:+.3f}",
                    "貢献度": f"{v['contribution']:+.4f}",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with viz3:
        try:
            fig3d = make_3d_scatter(standings, home, away)
            if fig3d.data:
                st.plotly_chart(
                    fig3d, width="stretch",
                    config={
                        "displayModeBar": True,
                        "modeBarButtonsToRemove": ["toImage", "sendDataToCloud"],
                        "scrollZoom": False,       # スマホでの誤ズーム防止
                        "responsive": True,        # 画面幅に自動フィット
                        "doubleClick": "reset",    # ダブルタップでリセット
                    },
                )
                st.caption(f"🔵 = {home}　🔴 = {away}　グレー = その他チーム（スワイプで回転）")
            else:
                st.info("順位表データなし")
        except Exception as e:
            st.info(f"3Dグラフ描画スキップ: {e}")


# ─── ワンボタン予測パイプライン ────────────────────────────

def _classify_prediction(pred: dict, closeness: float = 0.5) -> dict:
    """予測の確信度とdraw警戒を判定"""
    h = int(pred.get("home_win_prob", 40))
    d = int(pred.get("draw_prob", 25))
    a = int(pred.get("away_win_prob", 35))
    mx = max(h, d, a)
    if mx >= 50:
        confidence_level = "high"
    elif mx >= 40:
        confidence_level = "medium"
    else:
        confidence_level = "low"
    draw_alert = d >= 25 and closeness >= 0.5
    return {"confidence_level": confidence_level, "draw_alert": draw_alert,
            "max_prob": mx, "closeness": closeness}


def _render_spotlight(valid_preds: list[dict]):
    """今日の注目試合セクション: 堅い/波乱/要注意の3分類 + 理由付き"""
    if not valid_preds:
        return

    solid, upset, caution = [], [], []
    for p in valid_preds:
        pred = p.get("prediction", {})
        cls = p.get("classification", {})
        mx = cls.get("max_prob", 0)
        da = cls.get("draw_alert", False)
        closeness = cls.get("closeness", 0.5)
        home = p["match"]["home"]
        away = p["match"]["away"]
        label = f"{home} vs {away}"
        h = int(pred.get("home_win_prob", 40))
        d = int(pred.get("draw_prob", 25))
        a = int(pred.get("away_win_prob", 35))
        match_key = f"{home}_{away}"

        if mx >= 50:
            fav = home if h >= a else away
            reason = f"{fav}優勢 (確率{mx}%)"
            solid.append((label, h, d, a, reason))
            p["_spotlight"] = "solid"
        elif da:
            reason = f"実力接近 (接近度{closeness:.2f})"
            caution.append((label, h, d, a, reason))
            p["_spotlight"] = "caution"
        elif abs(h - a) <= 8:
            reason = f"僅差 (H{h}% vs A{a}%)"
            upset.append((label, h, d, a, reason))
            p["_spotlight"] = "upset"

    if not solid and not upset and not caution:
        return

    def _pill(items, emoji, title, bg, border, color):
        if not items:
            return ""
        cards = ""
        for name, h, d, a, reason in items[:3]:
            cards += (
                f'<div style="font-size:0.7rem;padding:2px 0;">'
                f'{name} <span style="color:#64748b;">({h}-{d}-{a})</span><br>'
                f'<span style="font-size:0.6rem;color:{color}99;">{reason}</span></div>'
            )
        return (
            f'<div style="flex:1;min-width:180px;padding:0.5rem 0.7rem;'
            f'background:{bg};border:1px solid {border};border-radius:8px;">'
            f'<div style="font-size:0.75rem;font-weight:700;color:{color};margin-bottom:0.3rem;">'
            f'{emoji} {title}</div>{cards}</div>'
        )

    html = (
        '<div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin:0.5rem 0 0.6rem;">'
        + _pill(solid, "🔒", "堅い試合", "#f0fdf4", "#bbf7d0", "#15803d")
        + _pill(upset, "⚡", "波乱の可能性", "#fef2f2", "#fecaca", "#dc2626")
        + _pill(caution, "⚠", "引分け要注意", "#fefce8", "#fef08a", "#a16207")
        + '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _fetch_step(label: str, fn, progress_bar, step: int, total: int):
    """データ取得ステップを実行し結果を返す。失敗時はNone。"""
    progress_bar.progress(step / total, text=f"{label}...")
    try:
        result = fn()
        return {"success": True, "data": result, "label": label}
    except Exception as e:
        return {"success": False, "data": None, "label": label, "error": str(e)}


def render_onebutton(division: str):
    """ワンボタン予測タブのメインUI"""
    cache_key = f"onebutton_{division}"

    # ── 主ボタン ──
    st.markdown("""
    <div style="text-align:center;padding:0.5rem 0;">
      <div style="font-size:0.85rem;color:#64748b;">
        ボタンを押すと最新データ取得 → ELO更新 → 全試合予測を自動実行します
      </div>
    </div>""", unsafe_allow_html=True)

    run_btn = st.button(
        "🚀 最新データ更新して予測する",
        type="primary", use_container_width=True,
        disabled=st.session_state.get("_onebutton_running", False),
    )

    if run_btn:
        st.session_state["_onebutton_running"] = True
        try:
            _run_onebutton_pipeline(division, cache_key)
        except Exception as _pipe_err:
            st.session_state["_onebutton_running"] = False
            st.error(f"パイプライン実行中にエラーが発生しました: {_pipe_err}")
            import traceback as _tb
            st.code(_tb.format_exc())
            return
        finally:
            st.session_state["_onebutton_running"] = False

    # ── 結果表示 ──
    if cache_key not in st.session_state:
        st.markdown("""
        <div style="text-align:center;padding:3rem 0;color:#374151;">
          <div style="font-size:3rem;">🚀</div>
          <div style="margin-top:0.8rem;font-size:0.95rem;color:#64748b;">
            上のボタンを押すと<br>最新データの取得から全試合予測まで自動実行されます
          </div>
        </div>""", unsafe_allow_html=True)
        return

    result = st.session_state[cache_key]
    _render_onebutton_results(result, division)


def _run_onebutton_pipeline(division: str, cache_key: str):
    """ワンボタン予測パイプライン: data_connectorで取得→予測→保存"""
    from datetime import datetime as _dt
    from data_connector import run_data_pipeline, build_feature_snapshot, compute_data_quality

    progress = st.progress(0.0, text="準備中...")

    def _progress_cb(step, total, label):
        progress.progress(step / (total + 2), text=label)

    # ── データ取得 (data_connector に委譲) ──
    pipeline = run_data_pipeline(division, progress_cb=_progress_cb)

    matches = pipeline.fixtures.data or []
    standings = pipeline.standings.data if pipeline.standings.success else pd.DataFrame()
    xg_data = pipeline.xg.data or {}
    cards_data = pipeline.discipline.data or {}
    past = pipeline.results.data or []
    elo_ratings = pipeline.elo.data or {}

    if not matches:
        progress.empty()
        st.warning("試合スケジュールが取得できませんでした")
        return

    # ELOスコア計算ヘルパー
    def _elo_scores(home: str, away: str) -> tuple[float, float]:
        initial, hb = 1500.0, 50.0
        rh = elo_ratings.get(home, initial) + hb
        ra = elo_ratings.get(away, initial)
        eh = 1.0 / (1.0 + 10.0 ** ((ra - rh) / 400.0))
        return eh, 1.0 - eh

    def _row(team: str) -> dict:
        if isinstance(standings, pd.DataFrame) and not standings.empty:
            r = standings[standings["チーム"] == team]
            return r.iloc[0].to_dict() if not r.empty else {}
        return {}

    # ── 全試合予測 ──
    total_p = len(matches)
    preds = []
    snapshots = []
    for i, match in enumerate(matches):
        home, away = match["home"], match["away"]
        progress.progress((7 + (i / total_p)) / (8 + 1),
                          text=f"予測中 {i+1}/{total_p}: {home} vs {away}")
        try:
            home_stats = _row(home)
            away_stats = _row(away)
            home_form = cached_form(home)
            away_form = cached_form(away)
            h2h = cached_h2h(home, away)
            home_inj = get_injury_news(home)
            away_inj = get_injury_news(away)
            home_venue = get_venue_info(home, match.get("venue"))
            away_venue = get_venue_info(away)
            weather = cached_weather(home_venue["lat"], home_venue["lon"], match["date"])
            home_xg = xg_data.get(home, {})
            away_xg = xg_data.get(away, {})
            home_cards = cards_data.get(home, {})
            away_cards = cards_data.get(away, {})
            home_days = calc_match_interval(home, match["date"], past)
            away_days = calc_match_interval(away, match["date"], past)
            elo_h, elo_a = _elo_scores(home, away)

            contributions = calculate_parameter_contributions(
                home, away, home_stats, away_stats, home_form, away_form,
                h2h, weather, home_inj, away_inj, home_venue, away_venue,
                home_xg=home_xg, away_xg=away_xg,
                home_cards=home_cards, away_cards=away_cards,
                home_days=home_days, away_days=away_days,
                elo_home_score=elo_h, elo_away_score=elo_a,
            )
            # v7 predict (baseline/Gemini reasoning生成)
            v7_prediction = predict_with_gemini(
                home, away, contributions,
                home_stats, away_stats,
                home_form, away_form, h2h, weather,
                home_xg=home_xg, away_xg=away_xg,
                home_days=home_days, away_days=away_days,
                home_cards=home_cards, away_cards=away_cards,
                home_injuries=home_inj, away_injuries=away_inj,
            )
            closeness = contributions.get("closeness", 0.5)

            # Primary model selection (hybrid_v9.1 or v7 fallback)
            from scripts.predict_logic import PRIMARY_MODEL_VERSION, compute_hybrid_v9, compute_shadow_v8_1
            if PRIMARY_MODEL_VERSION == "hybrid_v9.1":
                try:
                    hybrid = compute_hybrid_v9(
                        home, away, home_stats, away_stats,
                        home_form, away_form,
                        v7_prediction=v7_prediction,
                        elo_home_score=elo_h, elo_away_score=elo_a,
                        xg_home=home_xg, xg_away=away_xg,
                    )
                    # Primary = hybrid probs + v7のreasoning/score/confidence
                    prediction = dict(v7_prediction)
                    prediction["home_win_prob"] = hybrid["home_win_prob"]
                    prediction["draw_prob"] = hybrid["draw_prob"]
                    prediction["away_win_prob"] = hybrid["away_win_prob"]
                    prediction["hybrid_selection"] = hybrid["selection"]
                    prediction["skellam_raw"] = hybrid.get("skellam_raw", {})
                    prediction["skellam_boost"] = hybrid.get("skellam_boost", 0)
                    prediction["model_version"] = "hybrid_v9.1"
                    primary_version = "hybrid_v9.1"
                except Exception:
                    # Fallback to v7 if hybrid fails
                    prediction = v7_prediction
                    primary_version = "v7_refined"
            else:
                prediction = v7_prediction
                primary_version = "v7_refined"

            cls = _classify_prediction(prediction, closeness)
            gemini_used = "gemini" in str(v7_prediction.get("model", "")).lower()
            dq = compute_data_quality(pipeline, home_xg, away_xg, gemini_used)
            # 統計モデル事前確率 → 補正差分 (v7との差分を表示)
            stat_h, stat_d, stat_a = advantage_to_probs(
                contributions["raw_home_advantage"], closeness)
            # primary vs statの差分
            gem_diff = None
            if gemini_used or primary_version == "hybrid_v9.1":
                gem_diff = {
                    "home": int(prediction.get("home_win_prob", stat_h)) - stat_h,
                    "draw": int(prediction.get("draw_prob", stat_d)) - stat_d,
                    "away": int(prediction.get("away_win_prob", stat_a)) - stat_a,
                }
            preds.append({
                "match": match, "prediction": prediction,
                "home_form": home_form, "away_form": away_form,
                "classification": cls, "gemini_used": gemini_used,
                "data_quality": dq,
                "stat_prior": {"home": stat_h, "draw": stat_d, "away": stat_a},
                "gemini_diff": gem_diff,
            })
            snapshots.append(build_feature_snapshot(
                match, prediction, contributions, pipeline, gemini_used,
            ))
            # Shadow v8.1 併走予測
            try:
                shadow_pred = compute_shadow_v8_1(
                    home, away, home_stats, away_stats,
                    home_form, away_form,
                    elo_home_score=elo_h, elo_away_score=elo_a,
                )
            except Exception:
                shadow_pred = None
            # 保存: primary(hybrid), baseline(v7), shadow(v8.1)
            store_save(
                division, match, prediction,
                shadow_prediction=shadow_pred,
                baseline_prediction=v7_prediction,
                model_version=primary_version,
                baseline_model_version="v7_refined",
            )
        except Exception as exc:
            import traceback as _tb
            preds.append({
                "match": match,
                "error": str(exc),
                "error_trace": _tb.format_exc(),
                "classification": {"confidence_level": "low", "draw_alert": False},
            })

    progress.progress(1.0, text="完了!")
    progress.empty()

    # エラー率チェック: 全試合がエラーなら明示的に報告
    n_errors = sum(1 for p in preds if "error" in p)
    if n_errors == len(preds) and preds:
        first_err = preds[0]
        st.error(f"全 {len(preds)} 試合で予測エラーが発生しました")
        st.error(f"エラー: {first_err.get('error', 'unknown')}")
        trace = first_err.get("error_trace", "")
        if trace:
            with st.expander("スタックトレース", expanded=False):
                st.code(trace)
        # cache_keyはセットせずに return
        return
    elif n_errors > 0:
        st.warning(f"{n_errors}/{len(preds)} 試合で予測エラーが発生しました (他は正常)")

    st.session_state[cache_key] = {
        "preds": preds,
        "standings": standings,
        "pipeline_summary": pipeline.source_summary,
        "skipped": pipeline.skipped_names,
        "fetched_at": pipeline.generated_at,
        "n_matches": len(matches),
        "snapshots": snapshots,
    }
    st.rerun()


def _render_onebutton_results(result: dict, division: str):
    """ワンボタン予測結果の表示"""
    preds = result["preds"]
    standings = result.get("standings", pd.DataFrame())
    fetched_at = result.get("fetched_at", "?")
    skipped = result.get("skipped", [])

    # ── 鮮度パネル ──
    time_display = fetched_at[:16].replace("T", " ") if len(fetched_at) >= 16 else fetched_at
    source_badges = ""
    for item in result.get("pipeline_summary", []):
        c = item.get("color", "#64748b")
        source_badges += (
            f'<span style="font-size:0.68rem;padding:2px 8px;margin:1px;'
            f'background:{c}15;border:1px solid {c}55;border-radius:999px;color:{c};">'
            f'{item["name"]}: {item["badge"]}'
            f'</span>'
        )
    st.markdown(f"""
    <div style="margin-bottom:0.8rem;">
      <div style="display:flex;gap:0.4rem;flex-wrap:wrap;align-items:center;margin-bottom:0.4rem;">
        <span style="font-size:0.72rem;padding:3px 10px;background:#f0fdf4;border:1px solid #bbf7d0;
                     border-radius:999px;color:#15803d;">
          更新: {time_display}
        </span>
        <span style="font-size:0.72rem;padding:3px 10px;background:#eff6ff;border:1px solid #bfdbfe;
                     border-radius:999px;color:#1d4ed8;">
          {result['n_matches']}試合
        </span>
      </div>
      <div style="display:flex;gap:0.3rem;flex-wrap:wrap;">
        {source_badges}
      </div>
    </div>""", unsafe_allow_html=True)

    # ── サマリーカウント ──
    valid = [p for p in preds if "error" not in p]
    n_high = sum(1 for p in valid if p["classification"]["confidence_level"] == "high")
    n_mid = sum(1 for p in valid if p["classification"]["confidence_level"] == "medium")
    n_low = sum(1 for p in valid if p["classification"]["confidence_level"] == "low")
    n_draw = sum(1 for p in valid if p["classification"]["draw_alert"])

    # 直近予測精度 (prediction_store から)
    past_preds = store_load_all()
    past_stats = get_accuracy_stats(past_preds)
    acc_text = ""
    acc_n = past_stats.get("with_actual", 0)
    if past_stats.get("accuracy") is not None:
        acc_pct = round(past_stats["accuracy"] * 100, 1)
        acc_text = f"{acc_pct}%"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("高確信", f"{n_high}試合", help="max_prob >= 50%")
    c2.metric("中確信", f"{n_mid}試合", help="max_prob 40-50%")
    c3.metric("低確信", f"{n_low}試合", help="max_prob < 40%")
    c4.metric("Draw警戒", f"{n_draw}試合", help="draw >= 25% かつ closeness >= 0.5")
    c5.metric("直近正答率", acc_text or "--",
              delta=f"n={acc_n}" if acc_n else None,
              delta_color="off",
              help="成績記録タブで結果登録後に表示")

    # ── 今日の注目試合 ──
    _render_spotlight(valid)

    # ── 品質ランク凡例 (常設) ──
    st.markdown("""
    <div style="display:flex;gap:0.6rem;flex-wrap:wrap;align-items:center;
                padding:0.5rem 0.8rem;margin:0.4rem 0 0.6rem;
                background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;">
      <span style="font-size:0.7rem;color:#475569;font-weight:600;">データ品質:</span>
      <span style="font-size:0.65rem;padding:1px 6px;background:#dcfce7;color:#15803d;
                   border-radius:4px;border:1px solid #86efac;font-weight:700;">A</span>
      <span style="font-size:0.63rem;color:#64748b;">全ソース</span>
      <span style="font-size:0.65rem;padding:1px 6px;background:#dbeafe;color:#2563eb;
                   border-radius:4px;border:1px solid #93c5fd;font-weight:700;">B</span>
      <span style="font-size:0.63rem;color:#64748b;">高品質</span>
      <span style="font-size:0.65rem;padding:1px 6px;background:#fef9c3;color:#a16207;
                   border-radius:4px;border:1px solid #fde047;font-weight:700;">C</span>
      <span style="font-size:0.63rem;color:#64748b;">公式中心</span>
      <span style="font-size:0.65rem;padding:1px 6px;background:#fee2e2;color:#dc2626;
                   border-radius:4px;border:1px solid #fca5a5;font-weight:700;">D</span>
      <span style="font-size:0.63rem;color:#64748b;">データ不足</span>
      <span style="font-size:0.5rem;color:#94a3b8;margin-left:0.3rem;">|</span>
      <span style="font-size:0.7rem;color:#475569;font-weight:600;">確信度:</span>
      <span style="font-size:0.63rem;padding:2px 7px;background:#dcfce7;color:#15803d;
                   border-radius:999px;border:1px solid #86efac;">高確信</span>
      <span style="font-size:0.63rem;padding:2px 7px;background:#fef9c3;color:#a16207;
                   border-radius:999px;border:1px solid #fde047;">中確信</span>
      <span style="font-size:0.63rem;padding:2px 7px;background:#fee2e2;color:#dc2626;
                   border-radius:999px;border:1px solid #fca5a5;">低確信</span>
    </div>""", unsafe_allow_html=True)

    # ── フィルタ ──
    filter_opt = st.radio(
        "フィルタ", ["全件", "高確信", "Draw警戒", "低確信"],
        horizontal=True, label_visibility="collapsed",
    )
    if filter_opt == "高確信":
        preds_f = [p for p in preds if p.get("classification", {}).get("confidence_level") == "high"]
    elif filter_opt == "Draw警戒":
        preds_f = [p for p in preds if p.get("classification", {}).get("draw_alert")]
    elif filter_opt == "低確信":
        preds_f = [p for p in preds if p.get("classification", {}).get("confidence_level") == "low"]
    else:
        preds_f = list(preds)

    if not preds_f:
        st.info("該当する試合がありません")
        return

    # ── ソート: 高確信 → draw警戒 → その他 ──
    def _sort_key(p):
        cl = p.get("classification", {}).get("confidence_level", "low")
        da = p.get("classification", {}).get("draw_alert", False)
        # 高確信=0, draw警戒(非高確信)=1, 中確信=2, 低確信=3
        if cl == "high":
            return (0, 0)
        if da:
            return (1, 0)
        if cl == "medium":
            return (2, 0)
        return (3, 0)

    preds_f.sort(key=_sort_key)

    # ── Dランク試合があれば再取得ボタン ──
    n_d = sum(1 for p in preds_f if p.get("data_quality", {}).get("rank") == "D")
    if n_d > 0:
        st.warning(f"{n_d}試合でデータ品質がDランクです。再取得で改善する可能性があります。")
        if st.button("🔄 再取得して再予測", key="retry_d_rank"):
            cache_key = f"onebutton_{division}"
            st.session_state.pop(cache_key, None)
            st.rerun()

    # ── カード表示 ──
    for i in range(0, len(preds_f), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(preds_f):
                with col:
                    _render_enhanced_card(preds_f[i + j], standings)


def _render_enhanced_card(data: dict, standings: pd.DataFrame):
    """確信度/draw警戒バッジ付きの強化版予測カード"""
    match = data["match"]
    home, away = match["home"], match["away"]

    if "error" in data:
        st.markdown(f"""
        <div class="card">
          <div style="color:#f87171;font-size:0.8rem;">⚠ {home} vs {away} — 予測エラー</div>
          <div style="color:#64748b;font-size:0.72rem;">{data['error'][:80]}</div>
        </div>""", unsafe_allow_html=True)
        return

    pred = data["prediction"]
    cls = data.get("classification", {})
    import html as _html_mod
    h_pct = int(pred.get("home_win_prob", 40))
    d_pct = int(pred.get("draw_prob", 25))
    a_pct = int(pred.get("away_win_prob", 35))
    score = _html_mod.escape(str(pred.get("predicted_score", "?-?")))

    # 確信度バッジ
    cl = cls.get("confidence_level", "medium")
    cl_style = {
        "high": ("background:#dcfce7;color:#15803d;border-color:#86efac;", "高確信"),
        "medium": ("background:#fef9c3;color:#a16207;border-color:#fde047;", "中確信"),
        "low": ("background:#fee2e2;color:#dc2626;border-color:#fca5a5;", "低確信"),
    }.get(cl, ("", "?"))

    # draw警戒バッジ
    draw_badge = ""
    if cls.get("draw_alert"):
        draw_badge = ('<span style="font-size:0.68rem;padding:2px 8px;margin-left:4px;'
                      'background:#fef3c7;color:#92400e;border-radius:999px;'
                      'border:1px solid #fbbf24;">Draw警戒</span>')
    # Gemini / 統計モデル バッジ
    gemini_used = data.get("gemini_used", False)
    model_badge = (
        '<span style="font-size:0.6rem;padding:1px 6px;margin-left:3px;'
        'background:#ede9fe;color:#7c3aed;border-radius:999px;'
        'border:1px solid #c4b5fd;">Gemini</span>'
    ) if gemini_used else (
        '<span style="font-size:0.6rem;padding:1px 6px;margin-left:3px;'
        'background:#f0f9ff;color:#0284c7;border-radius:999px;'
        'border:1px solid #7dd3fc;">統計</span>'
    )

    # 勝者ハイライト
    if h_pct >= a_pct and h_pct >= d_pct:
        winner_icon = "🏠"
        home_w, away_w = "font-weight:900;color:#2563eb;", "color:#64748b;"
    elif a_pct > h_pct and a_pct >= d_pct:
        winner_icon = "✈"
        home_w, away_w = "color:#64748b;", "font-weight:900;color:#dc2626;"
    else:
        winner_icon = "🤝"
        home_w, away_w = "color:#64748b;", "color:#64748b;"

    # 順位
    def _rank(team):
        if isinstance(standings, pd.DataFrame) and not standings.empty:
            r = standings[standings["チーム"] == team]
            return f"({r.iloc[0]['順位']}位)" if not r.empty else ""
        return ""

    hf_html = form_html(data.get("home_form", []))
    af_html = form_html(data.get("away_form", []))

    # データ品質ランクバッジ
    dq = data.get("data_quality", {})
    dq_rank = dq.get("rank", "?")
    dq_color = dq.get("color", "#64748b")
    dq_label = _html_mod.escape(str(dq.get("label", "")))
    dq_note = _html_mod.escape(str(dq.get("note", "")))
    dq_sources = dq.get("sources_used", [])
    source_icons = " ".join(
        {"順位表": "📊", "試合結果": "📋", "ELO": "📈", "xG": "🎯",
         "規律": "🟨", "Gemini": "🤖"}.get(s, s) for s in dq_sources
    )

    # 品質ランクのツールチップ説明 (title属性用なのでHTMLエスケープ)
    import html as _html_mod
    dq_tooltip = _html_mod.escape({
        "A": "全ソース利用: 公式データ+xG+ELO+Gemini",
        "B": "高品質: 公式データ+ELO。xGまたはGeminiの片方が未使用",
        "C": "公式データ中心: xG・Gemini未使用の簡略版",
        "D": "データ不足: 順位表等の公式データが取得できていません",
    }.get(dq_rank, ""))

    # Dランク/Cランク時の警告バナー
    dq_missing = dq.get("missing", [])
    d_rank_warning = ""
    if dq_rank == "D":
        missing_text = "、".join(dq_missing) if dq_missing else "一部データ"
        d_rank_warning = (
            f'<div style="margin-top:0.4rem;padding:0.35rem 0.6rem;background:#fef2f2;'
            f'border:1px solid #fecaca;border-radius:6px;font-size:0.68rem;color:#991b1b;">'
            f'<b>不足:</b> {missing_text}<br>'
            f'参考度が低い予測です。上部の「🚀」ボタンで再取得をお試しください。'
            f'</div>'
        )
    elif dq_rank == "C":
        d_rank_warning = (
            '<div style="margin-top:0.4rem;padding:0.3rem 0.6rem;background:#fefce8;'
            'border:1px solid #fef08a;border-radius:6px;font-size:0.66rem;color:#854d0e;">'
            'xG・Gemini未使用の公式データ中心の簡略版予測です'
            '</div>'
        )

    # 高確信 × 低品質 (C/D) の矛盾警告
    if cl == "high" and dq_rank in ("C", "D"):
        d_rank_warning += (
            '<div style="margin-top:0.3rem;padding:0.3rem 0.6rem;background:#fff7ed;'
            'border:1px solid #fed7aa;border-radius:6px;font-size:0.66rem;color:#9a3412;">'
            '高確信ですがデータ品質が低いため、過信に注意してください'
            '</div>'
        )

    # データ品質明細 + AI分析（折りたたみ）
    import html as _html_mod
    reasoning = _html_mod.escape(str(pred.get("reasoning", "")))
    # Gemini補正の影響
    gem_diff = data.get("gemini_diff")
    stat_prior = data.get("stat_prior", {})
    gemini_diff_html = ""
    gemini_large_warning = ""
    if gem_diff and gemini_used:
        max_abs_diff = max(abs(gem_diff.get("home", 0)), abs(gem_diff.get("draw", 0)), abs(gem_diff.get("away", 0)))
        if max_abs_diff >= 10:
            gemini_large_warning = (
                '<div style="margin-top:0.3rem;padding:0.25rem 0.5rem;background:#faf5ff;'
                'border:1px solid #e9d5ff;border-radius:6px;font-size:0.64rem;color:#7c3aed;">'
                f'Geminiが統計モデルから大きく補正しています (最大{max_abs_diff}pp)'
                '</div>'
            )
        def _sign(v):
            return f"+{v}" if v > 0 else str(v)
        gemini_diff_html = (
            f'<br><b>Gemini補正:</b> '
            f'H {_sign(gem_diff["home"])}pp '
            f'D {_sign(gem_diff["draw"])}pp '
            f'A {_sign(gem_diff["away"])}pp '
            f'<span style="color:#94a3b8;">'
            f'(統計: {stat_prior.get("home","?")}%-{stat_prior.get("draw","?")}%-{stat_prior.get("away","?")}%)</span>'
        )

    details_inner = (
        f'<div style="font-size:0.66rem;color:#64748b;margin-top:0.3rem;">'
        f'<b>利用データ:</b> {", ".join(dq_sources) if dq_sources else "なし"}<br>'
        f'<b>品質:</b> {dq_label} — {dq_note}'
        f'{gemini_diff_html}'
        f'</div>'
    )
    if reasoning:
        details_inner += (
            f'<div style="font-size:0.68rem;color:#374151;margin-top:0.4rem;'
            f'line-height:1.5;border-top:1px solid #e5e7eb;padding-top:0.3rem;">{reasoning}</div>'
        )
    details_html = (
        f'{d_rank_warning}'
        f'{gemini_large_warning}'
        f'<details style="margin-top:0.4rem;">'
        f'<summary style="font-size:0.68rem;color:#64748b;cursor:pointer;">'
        f'{source_icons} 詳細を見る</summary>'
        f'{details_inner}'
        f'</details>'
    )

    # 注目試合ハイライト
    spotlight = data.get("_spotlight", "")
    spot_icon = {"solid": "🔒", "upset": "⚡", "caution": "⚠"}.get(spotlight, "")
    spot_border = {
        "solid": "border-left:3px solid #16a34a;",
        "upset": "border-left:3px solid #dc2626;",
        "caution": "border-left:3px solid #eab308;",
    }.get(spotlight, "")

    # カードHTML: ヘッダー+確率バー (固定長、reasoningを含まない)
    card_header = (
        f'<div class="card" style="margin-bottom:0.8rem;{spot_border}">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">'
        f'<span style="font-size:0.7rem;color:#4b5563;">'
        f'{spot_icon} {match["date"]} {match.get("time","?")} | {match.get("venue","?")}'
        f'</span>'
        f'<span>'
        f'<span style="font-size:0.68rem;padding:2px 8px;border-radius:999px;border:1px solid;{cl_style[0]}">{cl_style[1]}</span>'
        f'{draw_badge}{model_badge}'
        f'<span style="font-size:0.62rem;padding:1px 5px;margin-left:3px;'
        f'background:{dq_color}18;color:{dq_color};border-radius:4px;'
        f'border:1px solid {dq_color}55;font-weight:700;cursor:help;"'
        f' title="{dq_tooltip}">{dq_rank}</span>'
        f'</span></div>'
        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
        f'<div style="text-align:left;flex:1;">'
        f'<div style="font-size:1rem;{home_w}">🏠 {home}</div>'
        f'<div style="font-size:0.65rem;color:#4b5563;">{_rank(home)} {hf_html}</div>'
        f'</div>'
        f'<div style="text-align:center;padding:0 0.8rem;">'
        f'<div style="font-size:1.1rem;font-weight:900;color:#0f172a;">{winner_icon} {score}</div>'
        f'</div>'
        f'<div style="text-align:right;flex:1;">'
        f'<div style="font-size:1rem;{away_w}">{away} ✈</div>'
        f'<div style="font-size:0.65rem;color:#4b5563;">{_rank(away)} {af_html}</div>'
        f'</div></div>'
        f'<div style="border-radius:8px;overflow:hidden;display:flex;height:26px;margin-bottom:4px;">'
        f'<div style="width:{h_pct}%;background:#3b82f6;display:flex;align-items:center;justify-content:center;font-size:0.8rem;font-weight:700;color:white;">{h_pct}%</div>'
        f'<div style="width:{d_pct}%;background:#f59e0b;display:flex;align-items:center;justify-content:center;font-size:0.8rem;font-weight:700;color:white;">{d_pct}%</div>'
        f'<div style="width:{a_pct}%;background:#ef4444;display:flex;align-items:center;justify-content:center;font-size:0.8rem;font-weight:700;color:white;">{a_pct}%</div>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;font-size:0.62rem;color:#4b5563;">'
        f'<span>ホーム勝利</span><span>引き分け</span><span>アウェー勝利</span></div>'
        f'{_build_recommendation(cl, dq_rank, cls.get("draw_alert", False), h_pct, d_pct, a_pct)}'
        f'{details_html}'
        f'</div>'
    )
    st.markdown(card_header, unsafe_allow_html=True)


def _build_recommendation(
    confidence: str, dq_rank: str, draw_alert: bool,
    h_pct: int, d_pct: int, a_pct: int,
) -> str:
    """最終推奨バッジを生成。高確信=第一のみ、中/低確信=第一+第二。"""

    # ── 第一推奨 (argmax) ──
    probs = {"home": h_pct, "draw": d_pct, "away": a_pct}
    sorted_cls = sorted(probs.items(), key=lambda x: -x[1])
    first_cls, first_pct = sorted_cls[0]
    second_cls, second_pct = sorted_cls[1]

    first_label = {"home": "ホーム勝ち", "draw": "引き分け", "away": "アウェー勝ち"}[first_cls]

    # ── スタイル決定 ──
    if confidence == "high":
        style_icon, style_text = "🔒", "本命向き"
        style_bg, style_border, style_color = "#dcfce7", "#86efac", "#15803d"
    elif confidence == "medium":
        if dq_rank in ("A", "B") and not draw_alert:
            style_icon, style_text = "🎯", "組み合わせ向き"
            style_bg, style_border, style_color = "#eff6ff", "#bfdbfe", "#1d4ed8"
        elif draw_alert:
            style_icon, style_text = "⚡", "波乱狙い"
            style_bg, style_border, style_color = "#fefce8", "#fef08a", "#a16207"
        else:
            style_icon, style_text = "⏭", "スキップ推奨"
            style_bg, style_border, style_color = "#f1f5f9", "#cbd5e1", "#64748b"
    else:  # low
        style_icon, style_text = "⏭", "見送り推奨"
        style_bg, style_border, style_color = "#f1f5f9", "#cbd5e1", "#64748b"

    # 第一推奨バッジ
    first_html = (
        f'<span style="font-size:0.68rem;padding:2px 10px;'
        f'background:{style_bg};color:{style_color};border:1px solid {style_border};'
        f'border-radius:999px;">'
        f'{style_icon} {first_label} {first_pct}% — {style_text}</span>'
    )

    # ── 第二推奨 (中確信・低確信のみ) ──
    second_html = ""
    if confidence != "high":
        # draw警戒時はdrawを第二推奨に優先
        if draw_alert and second_cls != "draw" and d_pct >= second_pct - 3:
            second_cls = "draw"
            second_pct = d_pct

        second_label_map = {
            "draw": "引き分け寄り",
            "home": "勝敗寄り (ホーム勝ち)",
            "away": "勝敗寄り (アウェー勝ち)",
        }
        second_label = second_label_map.get(second_cls, "")
        second_html = (
            f'<span style="font-size:0.62rem;padding:1px 8px;margin-left:4px;'
            f'background:#f8fafc;color:#64748b;border:1px solid #e2e8f0;'
            f'border-radius:999px;">'
            f'2nd: {second_label} {second_pct}%</span>'
        )

    return (
        f'<div style="margin-top:0.35rem;text-align:right;">'
        f'{first_html}{second_html}'
        f'</div>'
    )


# ─── 全試合予測カード ─────────────────────────────────────

def _render_match_card(data: dict, standings: pd.DataFrame):
    """1試合分のコンパクト予測カードを描画"""
    match = data["match"]
    home  = match["home"]
    away  = match["away"]

    if "error" in data:
        st.markdown(f"""
        <div class="card">
          <div style="color:#f87171;font-size:0.8rem;">⚠️ {home} vs {away} — 予測エラー</div>
          <div style="color:#64748b;font-size:0.72rem;">{data['error'][:80]}</div>
        </div>""", unsafe_allow_html=True)
        return

    pred  = data["prediction"]
    h_pct = int(pred.get("home_win_prob", 40))
    d_pct = int(pred.get("draw_prob",     25))
    a_pct = int(pred.get("away_win_prob", 35))
    score = pred.get("predicted_score", "?-?")
    conf  = pred.get("confidence", "medium")

    conf_color = {"high": "#16a34a", "medium": "#ca8a04", "low": "#dc2626"}.get(conf, "#64748b")
    conf_label = {"high": "高 🟢", "medium": "中 🟡", "low": "低 🔴"}.get(conf, "?")

    # 勝者ハイライト
    if h_pct >= a_pct and h_pct >= d_pct:
        home_style = "font-weight:900;color:#2563eb;"
        away_style = "color:#64748b;"
    elif a_pct > h_pct and a_pct >= d_pct:
        home_style = "color:#64748b;"
        away_style = "font-weight:900;color:#dc2626;"
    else:
        home_style = "color:#64748b;"
        away_style = "color:#64748b;"

    # 順位取得
    def get_rank(team: str) -> str:
        if standings.empty:
            return ""
        r = standings[standings["チーム"] == team]
        return f"({r.iloc[0]['順位']}位)" if not r.empty else ""

    h_rank = get_rank(home)
    a_rank = get_rank(away)

    # フォームバッジ
    h_form_html = form_html(data.get("home_form", []))
    a_form_html = form_html(data.get("away_form", []))

    st.markdown(f"""
    <div class="card" style="margin-bottom:0.8rem;">
      <div style="font-size:0.7rem;color:#4b5563;margin-bottom:0.5rem;">
        📅 {match['date']} &nbsp; 🕐 {match.get('time','?')} &nbsp; 🏟️ {match.get('venue','?')}
      </div>

      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">
        <div style="text-align:left;flex:1;">
          <div style="font-size:1rem;{home_style}">🏠 {home}</div>
          <div style="font-size:0.65rem;color:#4b5563;">{h_rank} {h_form_html}</div>
        </div>
        <div style="text-align:center;padding:0 0.8rem;">
          <div style="font-size:1.1rem;font-weight:900;color:#0f172a;">⚽ {score}</div>
          <div style="font-size:0.6rem;color:#94a3b8;">予想スコア</div>
        </div>
        <div style="text-align:right;flex:1;">
          <div style="font-size:1rem;{away_style}">{away} ✈️</div>
          <div style="font-size:0.65rem;color:#4b5563;">{a_rank} {a_form_html}</div>
        </div>
      </div>

      <div style="border-radius:8px;overflow:hidden;display:flex;height:26px;margin-bottom:4px;">
        <div style="width:{h_pct}%;background:#3b82f6;display:flex;align-items:center;
                    justify-content:center;font-size:0.8rem;font-weight:700;color:white;">
          {h_pct}%
        </div>
        <div style="width:{d_pct}%;background:#f59e0b;display:flex;align-items:center;
                    justify-content:center;font-size:0.8rem;font-weight:700;color:white;">
          {d_pct}%
        </div>
        <div style="width:{a_pct}%;background:#ef4444;display:flex;align-items:center;
                    justify-content:center;font-size:0.8rem;font-weight:700;color:white;">
          {a_pct}%
        </div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:0.62rem;color:#4b5563;">
        <span>ホーム勝利</span><span>引き分け</span><span>アウェー勝利</span>
      </div>

      <div style="text-align:right;margin-top:0.4rem;">
        <span style="font-size:0.68rem;padding:2px 8px;
                     background:{conf_color}22;color:{conf_color};
                     border-radius:999px;border:1px solid {conf_color}55;">
          信頼度 {conf_label}
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_all_predictions(division: str):
    """全試合まとめ予測タブ"""
    cache_key = f"all_preds_{division}"

    with st.spinner("スケジュール取得中..."):
        matches = cached_matches(division)

    if not matches:
        st.warning("試合スケジュールが取得できませんでした。サイドバーの「データ更新」を試してください。")
        return

    st.markdown(f"**{division.upper()} 直近 {len(matches)} 試合**の一括予測")

    col_btn, col_clear, col_info = st.columns([2, 1, 4])
    with col_btn:
        run_all = st.button("🔮 全試合予測を実行", type="primary", use_container_width=True)
    with col_clear:
        if st.button("🗑️ リセット", use_container_width=True):
            st.session_state.pop(cache_key, None)
            st.rerun()
    with col_info:
        api_ok = bool(os.getenv("GEMINI_API_KEY", "").strip())
        if api_ok:
            st.caption(f"Gemini 2.5 Flash で {len(matches)} 試合を順次予測します（約 {len(matches)*2} 秒）")
        else:
            st.caption("Gemini 未接続 — 統計モデルで高速予測します")

    # ── 予測実行 ────────────────────────────────────────
    if run_all:
        with st.spinner("順位表・データ取得中..."):
            standings = cached_standings(division)

        def _row(team: str) -> dict:
            if standings.empty:
                return {}
            r = standings[standings["チーム"] == team]
            return r.iloc[0].to_dict() if not r.empty else {}

        preds: dict = {}
        progress = st.progress(0.0, text="予測準備中...")
        xg_data    = cached_fbref_xg(division)
        cards_data = cached_discipline(division)
        past       = cached_past_results(division)

        for i, match in enumerate(matches):
            home = match["home"]
            away = match["away"]
            label = f"予測中 {i+1}/{len(matches)}: {home} vs {away}"
            progress.progress((i) / len(matches), text=label)
            try:
                home_stats = _row(home)
                away_stats = _row(away)
                home_form  = cached_form(home)
                away_form  = cached_form(away)
                h2h        = cached_h2h(home, away)
                home_inj   = get_injury_news(home)
                away_inj   = get_injury_news(away)
                home_venue = get_venue_info(home, match.get("venue"))
                away_venue = get_venue_info(away)
                weather    = cached_weather(home_venue["lat"], home_venue["lon"], match["date"])
                home_xg    = xg_data.get(home, {})
                away_xg    = xg_data.get(away, {})
                home_cards = cards_data.get(home, {})
                away_cards = cards_data.get(away, {})
                home_days  = calc_match_interval(home, match["date"], past)
                away_days  = calc_match_interval(away, match["date"], past)
                elo_h, elo_a = get_elo_scores(division, home, away)

                contributions = calculate_parameter_contributions(
                    home, away,
                    home_stats, away_stats, home_form, away_form,
                    h2h, weather, home_inj, away_inj, home_venue, away_venue,
                    home_xg=home_xg, away_xg=away_xg,
                    home_cards=home_cards, away_cards=away_cards,
                    home_days=home_days, away_days=away_days,
                    elo_home_score=elo_h, elo_away_score=elo_a,
                )
                prediction = predict_with_gemini(
                    home, away, contributions,
                    home_stats, away_stats,
                    home_form, away_form, h2h, weather,
                    home_xg=home_xg, away_xg=away_xg,
                    home_days=home_days, away_days=away_days,
                    home_cards=home_cards, away_cards=away_cards,
                    home_injuries=home_inj, away_injuries=away_inj,
                )
                preds[f"{home}_vs_{away}"] = {
                    "match": match, "prediction": prediction,
                    "home_form": home_form, "away_form": away_form,
                }
            except Exception as exc:
                preds[f"{home}_vs_{away}"] = {"match": match, "error": str(exc)}

        progress.progress(1.0, text="完了！")
        progress.empty()
        st.session_state[cache_key] = preds
        st.session_state[f"{cache_key}_standings"] = standings

        # 予測履歴に自動保存
        saved_n = 0
        for data in preds.values():
            if "error" not in data:
                store_save(division, data["match"], data["prediction"])
                saved_n += 1
        st.success(f"✅ {saved_n}試合の予測を履歴に保存しました。「📈 成績記録」タブで実際の結果を入力できます。")

    # ── 結果表示 ────────────────────────────────────────
    if cache_key not in st.session_state:
        st.markdown("""
        <div style="text-align:center;padding:3rem 0;color:#374151;">
          <div style="font-size:2.5rem;">🗓️</div>
          <div style="margin-top:0.8rem;font-size:0.95rem;color:#64748b;">
            「全試合予測を実行」ボタンを押すと<br>全試合の予測がまとめて表示されます
          </div>
        </div>""", unsafe_allow_html=True)
        return

    preds    = st.session_state[cache_key]
    standings = st.session_state.get(f"{cache_key}_standings", pd.DataFrame())

    st.markdown("---")

    # サマリーテーブル
    summary_rows = []
    for data in preds.values():
        if "error" in data:
            continue
        m    = data["match"]
        pred = data["prediction"]
        h_p  = int(pred.get("home_win_prob", 40))
        d_p  = int(pred.get("draw_prob",     25))
        a_p  = int(pred.get("away_win_prob", 35))
        if h_p >= a_p and h_p >= d_p:
            winner = f"🏠 {m['home']}"
        elif a_p > h_p and a_p >= d_p:
            winner = f"✈️ {m['away']}"
        else:
            winner = "🤝 引き分け"
        summary_rows.append({
            "日付":     m["date"],
            "時刻":     m.get("time", "?"),
            "ホーム":   m["home"],
            "アウェー": m["away"],
            "予想スコア": pred.get("predicted_score", "?-?"),
            "ホーム%":  h_p,
            "引分%":    d_p,
            "アウェー%": a_p,
            "予測勝者": winner,
            "信頼度":   pred.get("confidence", "?"),
        })

    if summary_rows:
        with st.expander("📋 サマリーテーブル（全試合）", expanded=False):
            st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

    # カードグリッド（2カラム）
    items = list(preds.values())
    for i in range(0, len(items), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(items):
                with col:
                    _render_match_card(items[i + j], standings)


# ─── 今季バックテスト ─────────────────────────────────────

def run_season_backtest(division: str) -> tuple[int, int]:
    """
    今季の完了済み全試合を取得し、各試合を遡及予測して prediction_store に保存。
    実際の結果も自動登録。Returns: (保存件数, スキップ件数)
    """
    with st.spinner("今季の試合結果を取得中…"):
        past = get_past_results(division)

    if not past:
        st.warning("完了済み試合データが取得できませんでした。")
        return 0, 0

    with st.spinner("現在の順位表を取得中…"):
        standings = cached_standings(division)

    def _row(team: str) -> dict:
        if standings.empty:
            return {}
        r = standings[standings["チーム"] == team]
        return r.iloc[0].to_dict() if not r.empty else {}

    xg_data    = get_fbref_xg_stats(division)
    cards_data = get_team_discipline_stats(division)
    saved, skipped = 0, 0
    prog = st.progress(0.0, text="遡及予測を実行中…")

    for i, m in enumerate(past):
        prog.progress(i / len(past), text=f"予測中 {i+1}/{len(past)}: {m['home']} vs {m['away']}")
        try:
            home_stats  = _row(m["home"])
            away_stats  = _row(m["away"])
            home_form   = get_team_recent_form(m["home"])
            away_form   = get_team_recent_form(m["away"])
            h2h         = get_head_to_head(m["home"], m["away"])
            home_inj: list = []
            away_inj: list = []
            home_venue  = get_venue_info(m["home"], m.get("venue"))
            away_venue  = get_venue_info(m["away"])
            home_xg     = xg_data.get(m["home"], {})
            away_xg     = xg_data.get(m["away"], {})
            home_cards  = cards_data.get(m["home"], {})
            away_cards  = cards_data.get(m["away"], {})
            home_days   = calc_match_interval(m["home"], m["date"], past)
            away_days   = calc_match_interval(m["away"], m["date"], past)
            elo_h, elo_a = get_elo_scores(division, m["home"], m["away"])

            # ★ 実際の天気データ（アーカイブ）を使用
            weather = get_historical_weather(home_venue["lat"], home_venue["lon"], m["date"])

            contributions = calculate_parameter_contributions(
                m["home"], m["away"],
                home_stats, away_stats, home_form, away_form,
                h2h, weather, home_inj, away_inj, home_venue, away_venue,
                home_xg=home_xg, away_xg=away_xg,
                home_cards=home_cards, away_cards=away_cards,
                home_days=home_days, away_days=away_days,
                elo_home_score=elo_h, elo_away_score=elo_a,
            )

            # 統計モデルで確率算出（高速・再現性重視）
            h_pct, d_pct, a_pct = advantage_to_probs(
                contributions["raw_home_advantage"],
                contributions.get("closeness", 0.5),
            )

            if h_pct >= a_pct and h_pct >= d_pct:
                pred_winner = "home"
            elif a_pct > h_pct and a_pct >= d_pct:
                pred_winner = "away"
            else:
                pred_winner = "draw"

            prediction = {
                "home_win_prob":   h_pct,
                "draw_prob":       d_pct,
                "away_win_prob":   a_pct,
                "predicted_score": "—",      # 統計モデルはスコア予測なし
                "confidence":      "medium",
                "pred_winner":     pred_winner,
                "model":           "statistical-backtest",
            }

            # 予測を保存（同一試合は上書き）
            pred_id = store_save(division, m, prediction)

            # 実際の結果も即座に登録
            winner_label_map = {"home": "ホーム勝利", "draw": "引き分け", "away": "アウェー勝利"}
            store_update_actual(pred_id, m["score"], winner_label_map[m["winner"]])
            saved += 1

        except Exception as exc:
            logger.warning("Backtest failed for %s vs %s: %s", m["home"], m["away"], exc)
            skipped += 1

    prog.progress(1.0, text="完了！")
    prog.empty()
    return saved, skipped


# ─── 成績記録タブ ─────────────────────────────────────────

def render_history(division: str = "j1"):
    """予測履歴・実際の結果入力・正答率レポート"""

    # ── 今季振り返り分析ボタン ─────────────────────────────
    with st.expander("🔬 今季開幕からの遡及分析（バックテスト）", expanded=False):
        st.markdown(
            "今シーズンの全完了試合を**実際の天気データ**で遡及予測し、実際の結果と比較します。  \n"
            "※ 統計モデル（Gemini なし）で高速実行。予測は試合前情報のみで算出。"
        )
        col_bt, col_info = st.columns([2, 5])
        with col_bt:
            if st.button("⚡ 今季全試合を分析", type="primary", use_container_width=True):
                saved, skipped = run_season_backtest(division)
                if saved:
                    st.success(f"✅ {saved}試合を予測・登録しました（スキップ: {skipped}）")
                    st.rerun()
        with col_info:
            st.caption(
                "実行すると jleague.jp から試合結果を取得し、"
                "Open-Meteo アーカイブで実際の天気を参照して予測を生成します。"
            )

    st.markdown("---")

    preds = store_load_all()

    if not preds:
        st.markdown("""
        <div style="text-align:center;padding:3rem 0;">
          <div style="font-size:2.5rem;">📭</div>
          <div style="margin-top:0.8rem;color:#64748b;">
            まだ予測が保存されていません。<br>
            「🗓️ 全試合予測」タブで予測を実行すると自動的に保存されます。
          </div>
        </div>""", unsafe_allow_html=True)
        return

    # ── 正答率サマリー ─────────────────────────────────────
    stats = get_accuracy_stats(preds)
    acc_str = f"{stats['accuracy']:.1%}" if stats["accuracy"] is not None else "—"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📋 保存済み予測", stats["total"])
    c2.metric("⚽ 結果入力済み", stats["with_actual"])
    c3.metric("✅ 正解数", stats["correct"])
    c4.metric("🎯 正答率", acc_str)

    # 信頼度別正答率
    if stats["by_conf"]:
        with st.expander("信頼度別の正答率"):
            conf_rows = []
            for conf, v in stats["by_conf"].items():
                label = {"high": "高 🟢", "medium": "中 🟡", "low": "低 🔴"}.get(conf, conf)
                acc = f"{v['correct']/v['total']:.1%}" if v["total"] else "—"
                conf_rows.append({"信頼度": label, "予測数": v["total"],
                                   "正解": v["correct"], "正答率": acc})
            st.dataframe(pd.DataFrame(conf_rows), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 予測一覧・実際の結果入力")
    st.caption("「実際のスコア」「実際の勝者」列を編集して「💾 結果を保存」を押してください。")

    # ── データ準備 ─────────────────────────────────────────
    conf_label_map = {"high": "高", "medium": "中", "low": "低"}
    pred_winner_label_map = {"home": "ホーム勝利", "draw": "引き分け", "away": "アウェー勝利"}

    rows, id_list = [], []
    for p in preds:
        m    = p["match"]
        pred = p["prediction"]
        actual = p.get("actual") or {}

        pred_w_code  = pred.get("pred_winner", "home")
        pred_w_label = pred_winner_label_map.get(pred_w_code, "?")
        actual_w     = actual.get("winner_label", "")

        # 判定
        if actual_w:
            is_ok = actual.get("winner") == pred_w_code
            judge = "✅ 正解" if is_ok else "❌ 不正解"
        else:
            judge = "⏳ 未入力"

        id_list.append(p["id"])
        rows.append({
            "日付":       m["date"],
            "ホーム":     m["home"],
            "アウェー":   m["away"],
            "予測勝者":   pred_w_label,
            "予想スコア": pred.get("predicted_score", "?-?"),
            "信頼度":     conf_label_map.get(pred.get("confidence", ""), "?"),
            "実際のスコア": actual.get("score", ""),
            "実際の勝者":   actual_w,
            "判定":        judge,
        })

    df = pd.DataFrame(rows)

    edited_df = st.data_editor(
        df,
        column_config={
            "実際のスコア": st.column_config.TextColumn(
                "実際のスコア", help="例: 2-1", max_chars=10
            ),
            "実際の勝者": st.column_config.SelectboxColumn(
                "実際の勝者",
                options=["", "ホーム勝利", "引き分け", "アウェー勝利"],
            ),
        },
        disabled=["日付", "ホーム", "アウェー", "予測勝者", "予想スコア", "信頼度", "判定"],
        hide_index=True,
        use_container_width=True,
        height=min(60 + len(rows) * 36, 600),
    )

    col_save, col_del, col_space = st.columns([2, 2, 6])

    with col_save:
        if st.button("💾 結果を保存", type="primary", use_container_width=True):
            saved_n = 0
            for i, row in edited_df.iterrows():
                score  = str(row.get("実際のスコア", "") or "").strip()
                winner = str(row.get("実際の勝者",   "") or "").strip()
                if score or winner:
                    if store_update_actual(id_list[i], score, winner):
                        saved_n += 1
            st.success(f"{saved_n} 件の結果を保存しました")
            st.rerun()

    with col_del:
        with st.popover("🗑️ 削除…"):
            del_idx = st.selectbox(
                "削除する試合",
                options=range(len(id_list)),
                format_func=lambda i: f"{rows[i]['日付']} {rows[i]['ホーム']} vs {rows[i]['アウェー']}",
            )
            if st.button("この予測を削除", type="secondary"):
                store_delete(id_list[del_idx])
                st.success("削除しました")
                st.rerun()

    # ── 正解/不正解チャート ────────────────────────────────
    judged = [r for r in rows if r["判定"] != "⏳ 未入力"]
    if judged:
        st.markdown("---")
        st.markdown("#### 予測精度チャート")
        import plotly.express as px

        # 試合別判定
        judge_df = pd.DataFrame(judged)[["日付", "ホーム", "アウェー", "判定"]]
        judge_df["試合"] = judge_df["ホーム"] + " vs " + judge_df["アウェー"]
        correct_count = sum(1 for r in judged if r["判定"] == "✅ 正解")
        wrong_count   = len(judged) - correct_count

        pie_fig = px.pie(
            values=[correct_count, wrong_count],
            names=["正解", "不正解"],
            color_discrete_sequence=["#16a34a", "#dc2626"],
            hole=0.5,
        )
        pie_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(font=dict(color="#475569")),
            margin=dict(l=0, r=0, t=20, b=0), height=280,
        )
        pie_fig.update_traces(textfont_color="#ffffff")
        st.plotly_chart(pie_fig, width="stretch")

        # ── カテゴリ別正答率棒グラフ ──────────────────────────────
        analysis = analyze_predictions(preds)
        by_out = analysis.get("by_outcome", {})
        if any(v.get("accuracy") is not None for v in by_out.values()):
            outcome_labels = {"home": "ホーム勝利予測", "draw": "ドロー予測", "away": "アウェー勝利予測"}
            bar_data = [
                {"カテゴリ": outcome_labels.get(k, k),
                 "正答率(%)": round(v["accuracy"] * 100, 1) if v["accuracy"] is not None else 0,
                 "予測数": v["predicted"]}
                for k, v in by_out.items() if v["predicted"] > 0
            ]
            if bar_data:
                bar_fig = go.Figure(go.Bar(
                    x=[d["カテゴリ"] for d in bar_data],
                    y=[d["正答率(%)"] for d in bar_data],
                    text=[f"{d['正答率(%)']}%<br>({d['予測数']}試合)" for d in bar_data],
                    textposition="outside",
                    marker_color=["#3b82f6", "#f59e0b", "#ef4444"],
                ))
                bar_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(range=[0, 110], ticksuffix="%", gridcolor="#e2e8f0"),
                    xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                    margin=dict(l=0, r=0, t=10, b=0), height=260,
                    showlegend=False,
                )
                st.plotly_chart(bar_fig, use_container_width=True)

    # ── Gemini 今週の反省・改善案 ──────────────────────────────
    st.markdown("---")
    st.markdown("#### 🤖 Gemini 反省レポート & 重み調整提案")
    st.caption("不正解試合を Gemini 2.5 Flash が分析し、モデル改善案を提案します。")

    wrong_preds = [
        p for p in preds
        if p.get("actual") and p["actual"].get("winner")
        and p["actual"]["winner"] != p["prediction"].get("pred_winner")
    ]

    col_gen, col_sync = st.columns([2, 2])
    with col_gen:
        gen_disabled = len(wrong_preds) == 0
        if st.button(
            "🔍 Gemini で敗因分析を実行",
            type="primary",
            use_container_width=True,
            disabled=gen_disabled,
            help="不正解が1件以上必要です" if gen_disabled else "",
        ):
            with st.spinner("Gemini 2.5 Flash が分析中..."):
                report = ask_gemini_for_analysis(wrong_preds, preds, MODEL_WEIGHTS)
                st.session_state["feedback_report"] = report

    with col_sync:
        if st.button("🔄 実結果を自動同期", use_container_width=True,
                     help="jleague.jp から実スコアを取得し未入力予測に自動入力します"):
            with st.spinner("実結果を同期中..."):
                synced, skipped = sync_results_to_store(division)
            st.success(f"同期完了: {synced}件 入力 / {skipped}件 スキップ")
            st.rerun()

    # レポート表示
    report = st.session_state.get("feedback_report")
    if report:
        if report.get("error"):
            st.error(f"分析エラー: {report['error']}")
        else:
            gen_at = report.get("generated_at", "")[:16].replace("T", " ")
            st.caption(f"生成日時: {gen_at}")

            # 敗因
            causes = report.get("defeat_causes", [])
            if causes:
                st.markdown("**敗因分析**")
                for i, c in enumerate(causes, 1):
                    st.markdown(
                        f'<div style="background:#fef9c3;border-left:4px solid #f59e0b;'
                        f'padding:0.5rem 0.8rem;margin:0.3rem 0;border-radius:4px;'
                        f'font-size:0.9rem;color:#78350f;">⚠️ {i}. {c}</div>',
                        unsafe_allow_html=True,
                    )

            # 重み調整提案
            adjustments = report.get("weight_adjustments", [])
            if adjustments:
                st.markdown("**重み調整提案**")
                adj_rows = []
                for a in adjustments:
                    cur = a.get("current", 0)
                    sug = a.get("suggested", 0)
                    diff = sug - cur
                    arrow = "⬆️" if diff > 0 else "⬇️"
                    adj_rows.append({
                        "パラメータ": a.get("param", ""),
                        "現在値": f"{cur:.2f}",
                        "提案値": f"{sug:.2f}",
                        "変化": f"{arrow} {diff:+.2f}",
                        "理由": a.get("reason", ""),
                    })
                st.dataframe(pd.DataFrame(adj_rows), hide_index=True, use_container_width=True)

            # 新指標提案 + 自動実装ボタン
            new_inds = report.get("new_indicators", [])
            if new_inds:
                st.markdown("**新指標提案**")
                for ind in new_inds:
                    st.markdown(
                        f'<div style="background:#f0fdf4;border-left:4px solid #16a34a;'
                        f'padding:0.5rem 0.8rem;margin:0.3rem 0;border-radius:4px;font-size:0.9rem;">'
                        f'<strong style="color:#15803d;">💡 {ind.get("name","")}</strong>: '
                        f'{ind.get("description","")}'
                        f'<br><span style="color:#64748b;font-size:0.8rem;">'
                        f'期待効果: {ind.get("expected_impact","")}</span></div>',
                        unsafe_allow_html=True,
                    )

                # ── 自動実装ボタン ──────────────────────────────
                st.markdown("")
                if st.button(
                    "🚀 Gemini にこの指標を自動実装させる",
                    type="primary",
                    use_container_width=True,
                    key="btn_auto_implement",
                ):
                    with st.spinner("Gemini 2.5 Flash が実装コードを生成中..."):
                        impl = ask_gemini_to_implement_indicators(new_inds, MODEL_WEIGHTS)
                        st.session_state["impl_result"] = impl

            # 自動実装結果の表示
            impl = st.session_state.get("impl_result")
            if impl:
                st.markdown("---")
                st.markdown("#### ⚙️ 自動生成コード")
                if impl.get("error"):
                    st.error(f"生成エラー: {impl['error']}")
                else:
                    gen_at = impl.get("generated_at", "")[:16].replace("T", " ")
                    if impl.get("weight_normalized"):
                        st.warning("重みの合計が1.00でなかったため自動正規化しました。")
                    st.caption(f"生成日時: {gen_at}  |  {impl.get('summary','')}")

                    # スコア関数コード
                    funcs = impl.get("score_functions", [])
                    if funcs:
                        st.markdown("**① 新しいスコア関数 — `scripts/predict_logic.py` に追加**")
                        all_func_code = "\n\n".join(
                            f"# === {f.get('name','')} ===\n"
                            f"# データ: {f.get('data_note','')}\n"
                            f"{f.get('code','')}"
                            for f in funcs
                        )
                        st.code(all_func_code, language="python")

                    # 統合スニペット
                    snippet = impl.get("integration_snippet", "")
                    if snippet:
                        st.markdown("**② `calculate_parameter_contributions()` に追加する統合コード**")
                        st.code(snippet, language="python")

                    # 更新後の重み
                    uw = impl.get("updated_weights", {})
                    if uw:
                        st.markdown("**③ 更新後の `MODEL_WEIGHTS`**")
                        uw_lines = "MODEL_WEIGHTS: dict[str, float] = {\n"
                        for k, v in uw.items():
                            uw_lines += f'    "{k}": {v:.4f},\n'
                        uw_lines += "}"
                        st.code(uw_lines, language="python")

                        # ダウンロードボタン
                        patch_content = (
                            f"# Auto-generated by Gemini 2.5 Flash — {gen_at}\n"
                            f"# {impl.get('summary','')}\n\n"
                            f"# ━━━ ① スコア関数を predict_logic.py の既存関数の後に追加 ━━━\n\n"
                            f"{all_func_code if funcs else ''}\n\n"
                            f"# ━━━ ② calculate_parameter_contributions() に追加 ━━━\n\n"
                            f"{snippet}\n\n"
                            f"# ━━━ ③ MODEL_WEIGHTS を置き換え ━━━\n\n"
                            f"{uw_lines}\n"
                        )
                        st.download_button(
                            "📥 パッチファイルをダウンロード",
                            data=patch_content.encode("utf-8"),
                            file_name=f"predict_logic_patch_{gen_at[:10]}.py",
                            mime="text/x-python",
                            use_container_width=True,
                        )

            # 総評
            summary = report.get("summary", "")
            if summary:
                st.info(f"**総評**: {summary}")
    elif len(wrong_preds) == 0 and preds:
        st.success("不正解予測なし！すべて的中しています。")
    elif not preds:
        pass
    else:
        st.caption(f"不正解: {len(wrong_preds)}試合。「Gemini で敗因分析を実行」ボタンで分析できます。")


# ─── 順位表タブ ───────────────────────────────────────────

def render_standings(division: str):
    try:
        with st.spinner("順位表取得中..."):
            df = cached_standings(division)

        if df.empty:
            st.error("順位表データを取得できませんでした。")
            return

        # シーズン状態を試合数から判定してラベル表示
        max_games = int(df["試合"].max()) if "試合" in df.columns else 0
        is_pk_format = "PK勝" in df.columns
        has_groups = "グループ" in df.columns

        if division == "j1":
            if max_games <= 10:
                st.info(f"📊 J1 2026シーズン進行中（第{max_games}節終了時点）　※ EAST/WEST形式・PK決着あり")
            else:
                st.info(f"📊 J1 2026シーズン（{max_games}節終了時点）")
        elif division in ("j2", "j3"):
            st.info(f"📊 明治安田J2J3百年構想リーグ 2026（第{max_games}節終了時点）　※ 4グループ制・PK決着あり")

        # 表示用に数値型変換
        num_cols = ["順位", "試合", "勝点", "勝", "PK勝", "PK負", "分", "負", "得点", "失点"]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # 形式に応じた表示列
        if is_pk_format:
            col_order = ["順位", "チーム", "勝点", "試合", "勝", "PK勝", "PK負", "負",
                         "得点", "失点", "得失点差", "勝率"]
        else:
            col_order = ["順位", "チーム", "勝点", "試合", "勝", "分", "負",
                         "得点", "失点", "得失点差", "勝率"]

        def _render_standings_df(display_df: pd.DataFrame):
            styled = display_df.style
            try:
                styled = styled.background_gradient(subset=["勝点"], cmap="Blues")
                gd_num = pd.to_numeric(
                    display_df["得失点差"].astype(str).str.replace("+", "", regex=False),
                    errors="coerce"
                )
                styled = styled.background_gradient(gmap=gd_num, cmap="RdYlGn", subset=["得失点差"])
            except Exception:
                pass
            if "勝率" in display_df.columns:
                try:
                    styled = styled.format({"勝率": "{:.1%}"})
                except Exception:
                    pass
            n_rows = len(display_df)
            height = min(max(n_rows * 38 + 40, 200), 500)
            st.dataframe(styled, use_container_width=True, hide_index=True, height=height)

        if has_groups:
            # j2j3: グループ別に4タブで表示
            groups = df["グループ"].unique().tolist()
            tabs = st.tabs(groups)
            for tab, grp in zip(tabs, groups):
                with tab:
                    grp_df = df[df["グループ"] == grp].copy()
                    # グループ内で再ランク
                    grp_df = grp_df.sort_values(["勝点", "得失点差", "得点"], ascending=[False, False, False])
                    grp_df = grp_df.reset_index(drop=True)
                    grp_df["順位"] = range(1, len(grp_df) + 1)
                    display_cols = [c for c in col_order if c in grp_df.columns]
                    _render_standings_df(grp_df[display_cols].copy())
        else:
            display_cols = [c for c in col_order if c in df.columns]
            _render_standings_df(df[display_cols].copy())

    except Exception as exc:
        st.error(f"順位表の表示中にエラーが発生しました: {exc}")
        st.code(traceback.format_exc())


# ─── 使い方タブ ───────────────────────────────────────────

def render_about():
    try:
        st.markdown("## 使い方")

        st.markdown("### セットアップ")
        st.code(
            "pip install -r requirements.txt\n"
            "# .env に GEMINI_API_KEY=... を記入\n"
            "streamlit run app.py",
            language="bash",
        )

        st.markdown("### 予測モデル設計 v2（研究ベース最適化）")
        model_df = pd.DataFrame([
            ["チーム強度",    "18%", "勝点・順位差"],
            ["攻撃率",        "12%", "得点/試合 (Dixon-Coles λ) r≈0.78 vs 勝点"],
            ["守備率",        "10%", "失点/試合の逆数 (Dixon-Coles μ) r≈0.75 vs 勝点"],
            ["直近フォーム",  "22%", "PPG直近6試合（最信頼指標, 最適窓 Hvattum 2010）"],
            ["xG差分",        "10%", "期待ゴール差/試合 FBref J1 r≈0.87-0.92 vs 勝点"],
            ["ホームADV",     "12%", "J1ホーム勝率≈48% vs アウェー≈30%"],
            ["H2H",           " 7%", "Cohen's d≈0.2 (小効果, 直近5試合重み付け)"],
            ["選手状態",      " 5%", "怪我1人あたり-10%ペナルティ"],
            ["天気/疲労",     " 2%", "実測: 高温+0.4, 大雨+0.3 (効果≈+0.5%)"],
            ["移動距離",      " 2%", "Haversine + 疲労閾値 (J地理圧縮で最小化)"],
        ], columns=["パラメータ", "重み", "科学的根拠"])
        st.dataframe(model_df, hide_index=True, use_container_width=True)

        st.info(
            "**採用しなかった指標（研究が有効性を否定）**\n\n"
            "- ボール保有率: xG投入後の偏相関≈0、因果関係逆転問題あり (Lago-Ballesteros 2010)\n"
            "- パス成功率: チーム強度と共線形、独立した予測力なし\n"
            "- コーナー数: r≈0.25と最弱クラス\n"
            "- PPDA (プレス強度): J1公開データなし、効果+2.3%のみ (Robberechts 2019)"
        )

        st.markdown("### 移動距離と疲労換算")
        travel_df = pd.DataFrame([
            ["〜300km",      "0.00", "影響なし"],
            ["300〜600km",   "0.10", "軽微"],
            ["600〜1000km",  "0.25", "中程度（本州内長距離）"],
            ["1000〜1500km", "0.50", "大（本州〜北海道等）"],
            ["1500km超",     "0.75", "極大（沖縄等）"],
        ], columns=["移動距離", "疲労スコア", "説明"])
        st.dataframe(travel_df, hide_index=True, use_container_width=True)

        st.markdown("### 予測の読み方")
        st.info(
            "**ホーム勝利 %** → 青いカードに表示\n\n"
            "**引き分け %** → 黄色いカードに表示\n\n"
            "**アウェー勝利 %** → 赤いカードに表示\n\n"
            "3つの数字の合計は必ず100%になります。"
        )

    except Exception as exc:
        st.error(f"使い方ページの表示中にエラー: {exc}")


# ─── メイン ───────────────────────────────────────────────

def main():
    # Streamlit Cloud secrets → os.environ（毎回チェック、未設定時のみ転写）
    _apply_secrets()

    # PWA マニフェスト・モバイルメタタグを注入（初回のみ）
    if "pwa_injected" not in st.session_state:
        _inject_pwa_meta()
        st.session_state["pwa_injected"] = True

    # 正答率バッジ計算（ストアから）
    _all_preds = store_load_all()
    _stats = get_accuracy_stats(_all_preds)
    if _stats["accuracy"] is not None:
        _acc_pct  = round(_stats["accuracy"] * 100, 1)
        _acc_html = (
            f'<span style="display:inline-flex;align-items:center;gap:6px;'
            f'background:#f0fdf4;border:1.5px solid #16a34a;border-radius:999px;'
            f'padding:4px 14px;font-size:0.85rem;font-weight:700;color:#15803d;">'
            f'🎯 今季正答率&nbsp;<span style="font-size:1.1rem;">{_acc_pct}%</span>'
            f'&nbsp;<span style="font-size:0.72rem;font-weight:400;color:#4ade80;">({_stats["correct"]}/{_stats["with_actual"]}試合)</span>'
            f'</span>'
        )
    else:
        _acc_html = (
            '<span style="display:inline-flex;align-items:center;'
            'background:#f8fafc;border:1.5px solid #e2e8f0;border-radius:999px;'
            'padding:4px 14px;font-size:0.8rem;color:#94a3b8;">'
            '🎯 今季正答率&nbsp;—&nbsp;<span style="font-size:0.7rem;">（成績記録タブで分析を実行）</span>'
            '</span>'
        )

    st.markdown(f"""
    <div style="padding:1rem 0 0.6rem;display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:0.5rem;">
      <div>
        <div class="hero-title">⚽ サトシのJリーグ勝敗予測</div>
        <div class="hero-sub">Gemini 2.5 Flash × Open-Meteo × jleague.jp データ</div>
      </div>
      <div style="padding-top:0.3rem;">{_acc_html}</div>
    </div>
    """, unsafe_allow_html=True)

    division, match = sidebar()

    tab_one, tab_pred, tab_all, tab_hist, tab_stand, tab_about = st.tabs(
        ["🚀 ワンボタン予測", "🔮 個別予測", "🗓️ 全試合予測", "📈 成績記録", "📊 順位表", "ℹ️ 使い方"]
    )

    with tab_one:
        render_onebutton(division)

    with tab_pred:
        if match is None:
            st.markdown("""
            <div style="text-align:center;padding:4rem 0;color:#374151;">
              <div style="font-size:3rem;">⚽</div>
              <div style="margin-top:1rem;font-size:1rem;color:#64748b;">
                左上の <strong style="color:#3b82f6;">「＞」</strong> をタップしてサイドバーを開き<br>
                チームを選択して「予測を実行する」ボタンを押してください
              </div>
              <div style="margin-top:0.8rem;font-size:0.78rem;color:#94a3b8;">
                📱 スマホの方はサイドバーを開いてください
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            render_prediction(match, division)

    with tab_all:
        render_all_predictions(division)

    with tab_hist:
        render_history(division)

    with tab_stand:
        st.markdown(f"### {division.upper()} 順位表")
        render_standings(division)

    with tab_about:
        render_about()


if __name__ == "__main__":
    main()
