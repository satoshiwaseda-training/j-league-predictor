# ELO本番フロー設計書 (elo_production_flow.md)

> 作成日: 2026-04-08

---

## 1. データフロー

```
app.py起動
  |
  +-- cached_past_results(division) [TTL=1800s]
  |     → get_past_results() → 今季完了試合リスト
  |
  +-- cached_elo(division) [NEW, TTL=1800s]
  |     → build_elo_from_results(cached_past_results(division))
  |     → EloSystem オブジェクト (全試合反映済み)
  |
  +-- 予測時:
        → elo.score_pair(home, away) → (h_elo, a_elo)
        → calculate_parameter_contributions(..., elo_home=h_elo, elo_away=a_elo)
        → contributions["raw_home_advantage"] にELO寄与が加算
        → advantage_to_probs(raw_adv, closeness) → (h%, d%, a%)
```

---

## 2. 初期化タイミング

- **アプリ起動時**: `cached_past_results()` が呼ばれた時点でデータ取得
- **ELO構築**: `cached_elo()` 内で `cached_past_results()` を使い構築
- **TTL=1800秒**: 30分ごとにデータ再取得・ELO再構築
- **初回アクセス時のみ**: Streamlitのcache_dataにより自動管理

---

## 3. 更新タイミング

- ELOはキャッシュTTL満了時に再構築
- 試合結果確定後、get_past_results() が新結果を含む → 次回キャッシュ更新時にELO反映
- **リアルタイム更新ではない**: 最大30分の遅延はあるが、試合予測時にはTTL内で安定

---

## 4. リーク分析

| リスク | 評価 | 対策 |
|--------|------|------|
| 予測対象試合がELOに含まれる | **なし** | ELOは完了試合のみで構築。予測対象は未完了 |
| 未来試合の結果が混入 | **なし** | get_past_results() は "試合終了" フィルタあり |
| セッション間のstale state | **なし** | cache_data(TTL=1800) で自動更新 |
| 同一試合の再評価で値変動 | **なし** | ELOは cached_past_results() 依存で同一セッション内は不変 |

---

## 5. バックテストとの整合性

| 観点 | バックテスト | 本番 |
|------|------------|------|
| ELO構築 | results[0:idx] のみ | cached_past_results() 全件 |
| 更新順序 | 日付順 (厳密) | 日付順 (get_past_results() がソート済み) |
| 時系列保証 | index指定で厳密 | 完了試合のみ→OK |
| home_bonus | パラメータ化 | predict_logic.py の定数 |

---

## 6. 実装方針

### predict_logic.py
- `calculate_parameter_contributions()` に `elo_home_score`, `elo_away_score` 引数追加
  (デフォルト None → ELO未取得時は中立)
- MODEL_WEIGHTS に `"elo"` キーを追加 (最適化済み値)

### app.py
- `cached_elo(division)` 関数追加
- 予測呼び出し前に `elo = cached_elo(division)` → `elo.score_pair(home, away)` 取得
- `calculate_parameter_contributions()` に渡す

### Fallback
- ELO構築失敗時: elo_home_score=None → 中立値 (0.5, 0.5) で処理
- 過去試合が0件の場合: 全チーム初期値1500 → score_pair はホームボーナス分のみ

---

## 7. キャッシュ戦略

```python
@st.cache_data(ttl=1800, show_spinner=False)
def cached_elo(division: str) -> dict:
    """ELOレーティングを構築してスコアdictとして返す"""
    from scripts.predict_logic import EloSystem
    past = cached_past_results(division)
    elo = EloSystem(k=32.0, initial=1500.0, home_bonus=50.0)
    for r in past:
        elo.update(r["home"], r["away"], r["winner"])
    return {"_elo": elo, "n_matches": len(past)}
```

注意: `st.cache_data` は picklable な戻り値を要求するため、
EloSystem を直接返す代わりに ratings dict を返し、
予測時に score_pair 相当の計算を行う方式も検討。

→ **安全策**: EloSystem の ratings を dict として返し、
   predict_logic.py 側で score_pair を静的関数化する。
