# 本番統合設計書 (production_integration_plan.md)

> 作成日: 2026-04-08

---

## 1. 置き換え対象

### 旧関数: `advantage_to_probs(raw_advantage: float) -> tuple[int, int, int]`
- 場所: `scripts/predict_logic.py:731`
- 呼び出し元:
  1. `predict_logic.py:786` — `predict_with_gemini()` Gemini未設定fallback
  2. `predict_logic.py:964` — `predict_with_gemini()` Geminiエラーfallback
  3. `app.py:1280` — 統計モデル直接予測 (全試合一括予測タブ)

### 新関数: `advantage_to_probs(raw_advantage, closeness=0.5) -> tuple[int, int, int]`
- 同じ関数名で**シグネチャ互換維持** (`closeness`にデフォルト値を設定)
- 内部で3ロジット+softmaxを使用
- 旧呼び出し元は引数変更なしでそのまま動作

---

## 2. 変更箇所

### predict_logic.py への追加・変更

A. **EloSystem クラスの追加** (新規、ファイル末尾付近)
- backtest_runner.py から移植
- 本番では `get_past_results()` でELOを構築

B. **`advantage_to_probs()` の内部置き換え**
- 旧: sigmoid + 残差draw
- 新: 3ロジット + softmax
- `closeness` パラメータを追加 (デフォルト=0.5 で旧互換)

C. **`calculate_parameter_contributions()` の戻り値に `closeness` を追加**
- 既存の `raw_home_advantage` に加えて `closeness` を返す
- 呼び出し元は新フィールドを使うかどうかを選べる

### app.py への影響

- L1280: `advantage_to_probs(contributions["raw_home_advantage"])` → **変更不要**
  (closenessにデフォルト値があるため)
- 将来的に精度を上げたい場合:
  `advantage_to_probs(contributions["raw_home_advantage"], contributions.get("closeness", 0.5))`

---

## 3. rollback 方法

`advantage_to_probs` 内に `mode` 引数を残す:
```python
def advantage_to_probs(raw_advantage, closeness=0.5, mode="3logit"):
    if mode == "legacy":
        return _legacy_advantage_to_probs(raw_advantage)
    return _3logit_advantage_to_probs(raw_advantage, closeness)
```

rollback: `mode="legacy"` に切り替えるだけで旧挙動に戻る。

---

## 4. ELO データフロー

### 本番 (app.py)
```
app起動 → get_past_results() → EloSystem構築 → st.cache_data でキャッシュ
  → 各試合予測時に elo.score_pair(home, away) 参照
```

### バックテスト (backtest_runner.py)
```
get_multi_season_results() → 各試合indexまでのELO構築 → 予測
```

### 安全性
- ELOはキャッシュ (TTL=1800秒) で管理
- 更新はget_past_results()の結果のみ (未来リークなし)
- 同じ試合を複数回評価しても値は不変 (ELOはindex基準で構築)

---

## 5. UI互換性チェックポイント

| チェック項目 | 影響 |
|-------------|------|
| 個別予測: 確率表示 | 変更なし (h%, d%, a%) |
| 個別予測: レーダーチャート | 変更なし (contributions構造維持) |
| 個別予測: 貢献度チャート | 変更なし |
| 全試合予測: カードグリッド | 変更なし |
| 成績記録: 正答率 | 変更なし |
| 確率の合計=100% | softmax保証 |
