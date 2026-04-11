# 中確信0件の真因調査レポート (完全版)

> 調査日: 2026-04-11
> 対象: hybrid_v9.1 本番動作時の中確信カウント

---

## 1. 結論

**真因: 集計ロジックのバグではなく、hybrid_v9.1の出力分布と確信度閾値(40/50)のミスマッチ。**

- hybrid_v9.1 は **low (34-39%) か high (50%+) に2極化** する出力傾向
- 旧閾値 high>=50, medium 40-50, low<40 では medium 帯 (40-49%) の試合が少ない
- 10試合サンプルでは偶然 **medium=0** が発生しうる (約17%しか medium がない)

**修正**: 閾値を **47/37** に変更し、medium帯を広げる

---

## 2. 10試合の完全分類表 (修正前)

| 対戦カード | H% | D% | A% | max | closeness | draw_alert | 旧confidence | 新confidence | selection |
|-----------|----|----|----|----|-----------|------------|--------------|--------------|-----------|
| ジェフ千葉 vs 水戸 | 33 | 33 | 34 | 34 | 0.885 | ✓ | low | low | v7 |
| 町田 vs 柏 | 36 | 29 | 35 | 36 | 0.661 | ✓ | low | low | skellam |
| 横浜FM vs FC東京 | 33 | 34 | 33 | 34 | 0.944 | ✓ | low | low | v7 |
| 広島 vs 清水 | 33 | 35 | 32 | 35 | 0.960 | ✓ | low | low | v7 |
| 福岡 vs 長崎 | 28 | 33 | **39** | **39** | 0.670 | ✓ | low | **medium** | weighted |
| **神戸 vs 名古屋** | **63** | 21 | 16 | **63** | 0.531 | - | high | high | skellam |
| 京都 vs 岡山 | 34 | 35 | 31 | 35 | 0.960 | ✓ | low | low | v7 |
| **G大阪 vs C大阪** | **55** | 27 | 18 | **55** | 0.583 | ✓ | high | high | skellam |
| **浦和 vs 東京V** | **56** | 22 | 22 | **56** | 0.542 | - | high | high | skellam |
| 川崎F vs 鹿島 | 33 | 33 | 34 | 34 | 0.876 | ✓ | low | low | v7 |

### 分布の偏り

- 5試合 (50%): max 34-35% (v7採用, 完全接戦表示)
- 1試合 (10%): max 36-39% (weighted/skellam)
- 3試合 (30%): max 55-63% (Skellam採用, 高確信)
- **40-49%帯: 0試合** ← 旧閾値の "medium" ゾーンが空

---

## 3. confidence_class 判定条件

### 旧閾値 (修正前)
```python
if mx >= 50:   confidence_level = "high"
elif mx >= 40: confidence_level = "medium"  # ← 40-49
else:          confidence_level = "low"
```

### 新閾値 (修正後)
```python
if mx >= 47:   confidence_level = "high"
elif mx >= 37: confidence_level = "medium"  # ← 37-47 (広げた)
else:          confidence_level = "low"
```

### draw_alert は独立
```python
draw_alert = d >= 25 and closeness >= 0.5
```
**confidence_level と draw_alert は独立軸**。
1試合が high + draw_alert の両方にカウントされうる。

---

## 4. draw_alert 除外問題の確認

サマリーカウントコード (`_render_onebutton_results`):
```python
n_high = sum(1 for p in valid if p["classification"]["confidence_level"] == "high")
n_mid  = sum(1 for p in valid if p["classification"]["confidence_level"] == "medium")
n_low  = sum(1 for p in valid if p["classification"]["confidence_level"] == "low")
n_draw = sum(1 for p in valid if p["classification"]["draw_alert"])
```

**→ draw_alert はmedium集計から除外されていない**。
独立した別軸として数えている。

→ n_high + n_mid + n_low = n_valid (排他的)
→ n_draw は high/medium/low のいずれかと重複カウント

---

## 5. サマリーとカードの一致確認

各カードの confidence_class は `_classify_prediction` で計算し、
そのまま `preds[i]["classification"]["confidence_level"]` に保存。
サマリーも同じキーを参照するので **完全一致**。

矛盾は発生しない。

---

## 6. fallback 動作中の可能性

確認したところ、fallbackは発生していません。
- `PRIMARY_MODEL_VERSION = "hybrid_v9.1"` は正しくimport済み
- `compute_hybrid_v9()` は正常動作
- 選択ログ: v7=5, skellam=4, weighted=1

実際に hybrid_v9.1 が動いた結果として medium=0 が出ていた。

---

## 7. 閾値の統計的検証 (2025全377試合)

| 閾値 | high | medium | low |
|------|------|--------|-----|
| 50/40 (旧) | 27% | **17%** | 56% |
| **47/37 (採用)** | **31%** | **40%** | **30%** |
| 46/36 | 32% | 47% | 21% |
| 45/35 | 34% | 59% | 8% |
| 45/36 | 34% | 46% | 21% |

**47/37 が最もバランス良好** (30-40-30の均等配分に近い)。

---

## 8. 修正前後の件数 (10試合サンプル)

| 確信度 | 旧 | 新 |
|--------|----|----|
| high | 3 | 3 |
| medium | **0** | **1** |
| low | 7 | 6 |
| Draw警戒 | 8 | 8 |

### クロス集計 (新)
| 組み合わせ | 件数 |
|-----------|------|
| high + draw_alert | 1 |
| medium + draw_alert | 1 |
| low + draw_alert | 6 |

**medium (1件) と draw_alert (8件) は正しく重複カウント**。

---

## 9. 真因のまとめ

### a. 分類ロジックは正常
- 閾値判定は正しい
- draw_alert の独立性も正しい

### b. 集計は正常
- sum() は正しく動作
- medium 試合が 0 なら 0 と表示される

### c. 根本原因: hybrid_v9.1 の**出力分布**
- v7採用試合は closeness 接近時に `33-35%` 近傍に集中
- Skellam採用試合は `50%+` に集中
- **40-49% (旧medium ゾーン) が実データで少ない**
- 旧閾値は v7+Gemini 時代の分布前提だった

### d. 修正策
- 閾値を 47/37 に変更
- 全体の30-40-30分布に近づく
- 10試合でも medium=1件以上出やすくなる

---

## 10. 修正実装

`app.py: _classify_prediction()` (L980):
```python
# 旧: mx >= 50 → high, mx >= 40 → medium
# 新: mx >= 47 → high, mx >= 37 → medium
```

ヘルプテキストも更新:
- 高確信: "max_prob >= 47%"
- 中確信: "max_prob 37-47%"
- 低確信: "max_prob < 37%"

---

## 11. 閾値変更の必要性判断

| 項目 | 判定 |
|------|------|
| 集計ロジック変更 | **不要** (正常動作) |
| 確信度閾値変更 | **必要** (47/37に変更) |
| 中確信0件警告文言削除 | **推奨** (閾値変更で頻度低下) |

---

## 12. 今後の monitoring

新閾値適用後、予測実行ごとに:
- 高 ~31%, 中 ~40%, 低 ~30% が期待値
- 10試合サンプルで medium 3-4件程度が標準
- 0件が繰り返し発生する場合は hybrid_v9.1 の出力分布を再検証
