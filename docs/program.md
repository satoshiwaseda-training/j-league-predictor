# 実験計画書 (program.md)

> 作成日: 2026-04-08
> 目的: Jリーグ予測モデルの汎化性能改善を再現可能な方法で実施する

---

## 1. 目的関数

**主指標: accuracy (正答率)**
- 3クラス分類 (home / draw / away) の一致率
- 理由: 最もユーザーにとって直感的で、予測の「当たった/外れた」が明確

**副指標:**
- **macro F1**: クラス不均衡への対処評価 (drawは少数クラス)
- **log loss**: 確率キャリブレーションの評価 (自信過剰/過小の検出)
- **クラス別混同行列**: home/draw/awayそれぞれの精度と再現率
- **Brier score**: 確率の正確さの総合指標

---

## 2. データ分割方針

### 2.1 データソース
- **過去試合結果**: jleague.jpからスクレイプ可能な2024-2026シーズンの全試合
- **予測に使う特徴量**: 各試合時点の順位表・フォーム・xG等 (時系列厳守)

### 2.2 時系列分割 (未来情報リーク防止)

```
[====== Train ======][== Val ==][== Holdout ==]
     Season N          Season N     Season N+1
   (前半/前年)        (後半)       (翌年序盤)
```

**方式A: Walk-forward validation (推奨)**
- 各節ごとに「その節までのデータで訓練 → 次節を予測」
- 最もリアルな運用条件を再現
- シーズン内の全節を順に評価

**方式B: シーズン分割**
- Train: 2024シーズン全体
- Validation: 2025シーズン前半
- Holdout: 2025シーズン後半〜2026シーズン

### 2.3 リーク防止チェックリスト
- [ ] 順位表は該当試合**前**の節のデータを使用
- [ ] フォームは該当試合**前**の直近N試合を使用
- [ ] xGはシーズン累積ではなく該当試合**前**の累積を使用
- [ ] H2Hは該当試合**前**の対戦のみ使用

---

## 3. 停止条件

### 3.1 最適化ループ停止条件
以下の**いずれか**を満たしたとき停止:
1. **連続20回改善なし**: validation accuracyが20 trial連続で過去最良を更新しない
2. **最大試行回数**: 200 trial (上限)
3. **改善幅閾値**: 直近10回の改善幅が全て0.5pp未満
4. **過学習検知**: validation accuracy - train accuracy > 15pp

### 3.2 最終採用条件
以下の**全て**を満たした場合のみ本番反映:
1. validation accuracyがベースラインを上回る
2. holdout accuracyがベースラインと同等以上 (悪化なし)
3. log lossがベースラインより悪化していない
4. 既存の入出力インターフェースが維持されている
5. Streamlit表示に破壊なし

---

## 4. 探索対象

### 4.1 Phase 1: 構造簡素化 (死にパラメータ除去)
- 常にデフォルト値の6パラメータを除去
- 残り16パラメータの重みを再正規化
- **期待効果**: ノイズ除去による微小改善

### 4.2 Phase 2: 重みの最適化
- 有効16パラメータの重み配分を探索
- 制約: 合計=1.0, 各重み >= 0.0
- 探索手法: Optuna (TPE sampler)

### 4.3 Phase 3: スコア化関数の改善
- **フォーム期間**: 3, 4, 5, 6, 7, 8試合
- **フォーム重み減衰**: 線形 / 指数 / 均等
- **ホームアドバンテージ**: 固定値 vs チーム別学習値
- **シグモイド変換パラメータ**: K ∈ [1.0, 8.0], M ∈ [0.15, 0.45]

### 4.4 Phase 4: 新特徴量候補
- **ELOレーティング**: K=20-40, 初期値=1500, 試合ごと更新
- **引分傾向指標**: 得失点差の絶対値が小さいチーム同士 → draw率上昇
- **連戦疲労**: match_intervalの代わりに「直近N日でM試合」の密度指標
- **得点期待値差**: attack_rate_home - defense_rate_away の直接差分

---

## 5. 変更禁止領域

| 対象 | 理由 |
|------|------|
| `app.py` | Streamlit本番UI → 破壊禁止 |
| `data_fetcher.py` の公開API | 既存の呼び出し元に影響 |
| `predictor.py` のGemini統合 | LLM予測はバックテスト対象外 |
| `prediction_store.py` のデータ形式 | 既存データとの互換性 |
| `venues.py` | スタジアムDB → 変更不要 |

**変更可能**: `scripts/predict_logic.py` の重み・スコア化関数 (入出力IF維持)

---

## 6. 評価指標の計算方法

### 6.1 accuracy
```python
accuracy = correct_predictions / total_predictions
```

### 6.2 macro F1
```python
F1_per_class = 2 * precision * recall / (precision + recall)
macro_F1 = mean(F1_home, F1_draw, F1_away)
```

### 6.3 log loss
```python
log_loss = -mean(Σ y_true_c * log(y_pred_c) for c in {home, draw, away})
```
確率は [0.01, 0.99] にクリップ

### 6.4 Brier score
```python
brier = mean(Σ (y_pred_c - y_true_c)^2 for c in {home, draw, away})
```

---

## 7. 実験ログ仕様

各試行で以下を `experiment_logs.csv` に記録:

| フィールド | 型 | 内容 |
|-----------|------|------|
| timestamp | datetime | 実行日時 |
| experiment_id | str | 一意識別子 |
| phase | str | Phase 1/2/3/4 |
| parameter_set | json | 使用パラメータ(重み等) |
| features_used | json | 使用特徴量リスト |
| val_accuracy | float | validation accuracy |
| val_f1_macro | float | validation macro F1 |
| val_log_loss | float | validation log loss |
| holdout_accuracy | float | holdout accuracy (上位候補のみ) |
| holdout_f1_macro | float | holdout macro F1 |
| holdout_log_loss | float | holdout log loss |
| notes | str | 変更内容メモ |

---

## 8. 実験順序

```
Phase 1 (構造簡素化)
  → ベースライン測定
  → 死にパラメータ除去
  → 再評価
  ↓
Phase 2 (重み最適化)
  → Optuna探索
  → 上位候補をholdoutで検証
  ↓
Phase 3 (スコア化関数改善)
  → フォーム期間探索
  → シグモイド変換探索
  → ホームADV動的化
  ↓
Phase 4 (新特徴量)
  → ELO導入
  → 引分傾向指標
  → 最終holdout検証
```

各Phaseで改善が確認されたもののみ次Phaseに持ち越す。
