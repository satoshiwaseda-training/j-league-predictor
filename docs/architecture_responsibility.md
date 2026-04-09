# アーキテクチャ責務境界 (architecture_responsibility.md)

> 作成日: 2026-04-09

---

## レイヤー構成

```
[UI層]          app.py
                  |  render_onebutton() / render_prediction() / ...
                  |  UIイベント処理・表示・セッション管理
                  |
[統合ハブ]      data_connector.py
                  |  run_data_pipeline() / compute_data_quality() / build_feature_snapshot()
                  |  取得結果の統一(FetchResult)・品質判定・snapshot生成
                  |
[取得層]        data_fetcher.py
                  |  get_standings() / get_past_results() / get_upcoming_matches() / ...
                  |  個別ソースのスクレイピング・API呼び出し・パース
                  |
[予測層]        scripts/predict_logic.py
                  |  calculate_parameter_contributions() / advantage_to_probs() / predict_with_gemini()
                  |  特徴量スコア化・3ロジット変換・Gemini統合
                  |
[永続化層]      prediction_store.py
                  data/predictions.json
```

---

## 各ファイルの責務

### app.py (UI層)
- **する**: 画面描画、ユーザー操作処理、Streamlit session管理
- **しない**: HTTP通信、データパース、予測ロジック、品質判定

### data_connector.py (統合ハブ)
- **する**: data_fetcher.py の関数をFetchResult形式でラップ、
  パイプライン実行、データ品質ランク算出、feature snapshot生成
- **しない**: 画面描画、スクレイピング実装、予測計算

### data_fetcher.py (取得層)
- **する**: jleague.jp / fbref / Open-Meteo 等の個別ソースからデータ取得・パース
- **しない**: 品質判定、UI表示、予測計算

### scripts/predict_logic.py (予測層)
- **する**: 特徴量スコア化、3ロジットsoftmax変換、Geminiプロンプト生成・呼び出し
- **しない**: データ取得、UI表示

### prediction_store.py (永続化層)
- **する**: 予測履歴のJSON保存・読込・更新
- **しない**: それ以外

---

## データ品質ランク定義

| ランク | 条件 | UIバッジ色 |
|--------|------|-----------|
| A | 公式 + xG + ELO + Gemini (全ソース) | 緑 |
| B | 公式 + ELO + (xG or Geminiの片方) | 青 |
| C | 公式データのみ (xG/Geminiなし) | 黄 |
| D | 公式データも不完全 | 赤 |

---

## フロー図

```
[ユーザー] --click--> [app.py: render_onebutton]
  --> [data_connector: run_data_pipeline]
    --> [data_fetcher: get_upcoming_matches]  -> FetchResult(fixtures)
    --> [data_fetcher: get_past_results]      -> FetchResult(results)
    --> [data_fetcher: get_standings]         -> FetchResult(standings)
    --> [data_fetcher: get_fbref_xg_stats]   -> FetchResult(xg)
    --> [data_fetcher: get_team_discipline]   -> FetchResult(discipline)
    --> [predict_logic: EloSystem]            -> FetchResult(elo)
  <-- PipelineSnapshot
  --> [app.py: 各試合ループ]
    --> [predict_logic: calculate_parameter_contributions]
    --> [predict_logic: predict_with_gemini]
    --> [data_connector: compute_data_quality]  -> {rank, label, sources_used}
    --> [data_connector: build_feature_snapshot] -> {match_id, features, quality}
    --> [prediction_store: store_save]
  --> [app.py: _render_enhanced_card] (品質ランク+利用データ表示)
```
