# 確信度判定を diff ベースに変更

> 作成日: 2026-04-11

---

## 変更内容

### 旧ルール (max_prob ベース)
```python
mx = max(h, d, a)
if mx >= 47:   high
elif mx >= 37: medium
else:          low
```

### 新ルール (差分ベース)
```python
probs = sorted([h, d, a], reverse=True)
top1, top2 = probs[0], probs[1]
diff = top1 - top2
if diff >= 15: high
elif diff >= 5: medium
else:          low
```

---

## 変更理由

### 3クラス分類の本質
「確信度」の本質は「1位の選択肢が2位をどれだけ引き離しているか」。
max_prob 単独では以下の区別ができない:

| ケース | H | D | A | max | diff | 意味 |
|--------|---|---|---|-----|------|------|
| A | 40 | 35 | 25 | 40 | 5 | 弱い優勢 |
| B | 40 | 30 | 30 | 40 | 10 | 明確な優勢 |
| C | 50 | 25 | 25 | 50 | 25 | 強い優勢 |

旧ルール(max 47/37): A=medium, B=medium, C=high
**A (diff=5) と B (diff=10) が同じmediumだが、意味が異なる**。

新ルール(diff 15/5): A=medium, B=medium, C=high
**さらに A (diff=5) と C (diff=25) の区別が強化される**。

### 実データでの利点
hybrid_v9.1 は 2極化 (34-39% or 50%+) しやすい:
- 34-39%帯: max低いが、top2との差は1-5pp (拮抗)
- 50%+帯: max高く、top2との差も25-40pp (大差)

**max-basedでは接戦と大差を区別するしかないが、diff-basedでは**:
- max=34 / top2=33 → diff=1 → low (拮抗)
- max=50 / top2=25 → diff=25 → high (確信)
- max=45 / top2=30 → diff=15 → high (中→高)
- max=40 / top2=33 → diff=7 → medium (弱い優勢)

---

## 2025全体の分布比較

| 指標 | 旧 (max 47/37) | **新 (diff 15/5)** |
|------|---------------|-------------------|
| high | 31% | **33%** |
| medium | 40% | **28%** |
| low | 30% | **39%** |

両ルールとも概ねバランス良好。diff-basedは low が少し増える。

### 差分ヒストグラム (2025, n=377)
| diff帯 | 件数 | 割合 |
|--------|------|------|
| 0-2 | 103 | 27% |
| 3-5 | 67 | 18% |
| 6-8 | 59 | 16% |
| 9-11 | 12 | 3% |
| 12-14 | 13 | 3% |
| 15-17 | 11 | 3% |
| 18-20 | 11 | 3% |
| 21-23 | 9 | 2% |
| 24+ | 92 | 24% |

**45% の試合が diff<5** (拮抗)、**33%が diff>=15** (確信)、中間 28%。

---

## 現在10試合での比較 (2026-04-11付近)

| 対戦 | H | D | A | max | top2 | diff | 旧 | 新 | draw_alert |
|------|---|---|---|-----|------|------|-----|-----|------------|
| ジェフ vs 水戸 | 33 | 33 | 34 | 34 | 33 | 1 | low | low | ✓ |
| 町田 vs 柏 | 36 | 29 | 35 | 36 | 35 | 1 | low | low | ✓ |
| 横浜FM vs FC東京 | 33 | 34 | 33 | 34 | 33 | 1 | low | low | ✓ |
| 広島 vs 清水 | 33 | 35 | 32 | 35 | 33 | 2 | low | low | ✓ |
| 福岡 vs 長崎 | 28 | 33 | 39 | 39 | 33 | 6 | medium | medium | ✓ |
| **神戸 vs 名古屋** | **63** | 21 | 16 | 63 | 21 | **42** | high | high | - |
| 京都 vs 岡山 | 34 | 35 | 31 | 35 | 34 | 1 | low | low | ✓ |
| **G大阪 vs C大阪** | **55** | 27 | 18 | 55 | 27 | **28** | high | high | ✓ |
| **浦和 vs 東京V** | **56** | 22 | 22 | 56 | 22 | **34** | high | high | - |
| 川崎F vs 鹿島 | 33 | 33 | 34 | 34 | 33 | 1 | low | low | ✓ |

**この10試合は旧・新ともに同じ結果**: low=6, medium=1, high=3, draw_alert=8

理由: この10試合はhybrid_v9.1が**極端な2極化**しており、中間の弱優勢が1試合しかない。これは今節の特殊性で、2025全体では約28%が新ruleでmediumになる。

---

## 意思決定ロジックへの影響

### 推奨アクション変更(`_build_recommendation`)

| 確信度 | 表示 | 推奨 |
|--------|------|------|
| high + draw警戒 | 第一推奨のみ | 本命向き |
| high (draw警戒なし) | 第一推奨のみ | 本命向き |
| medium A/B品質 | 第一+第二推奨 | 組み合わせ向き |
| medium + draw警戒 | 第一+第二推奨 | 波乱狙い |
| medium C/D品質 | 第一+第二推奨 | スキップ推奨 |
| low | 第一+第二推奨 | 見送り推奨 |

判定ロジックは**confidence_level**のみ参照するため、
計算ロジック変更で自動的に整合。**コード変更不要**。

### カード左ボーダー (注目試合ハイライト)
spotlight 判定 (solid/upset/caution) は `confidence_level`, `draw_alert`, `|h-a|` を使用。
diff-based でもそのまま機能する。

### サマリーカウント
```python
n_high = sum(1 for p in valid if p["classification"]["confidence_level"] == "high")
n_mid  = sum(1 for p in valid if p["classification"]["confidence_level"] == "medium")
n_low  = sum(1 for p in valid if p["classification"]["confidence_level"] == "low")
n_draw = sum(1 for p in valid if p["classification"]["draw_alert"])
```
**変更不要**、新しい confidence_level が自動反映。

---

## 変更ファイル

### app.py: _classify_prediction()
- 引数 `pred, closeness` は変更なし
- 戻り値に `top1`, `top2`, `diff` を追加
- `confidence_level` は diff ベース計算
- `max_prob` は下位互換のため残す

### app.py: ヘルプテキスト更新
- 高確信: "top1-top2 差が15pp以上"
- 中確信: "top1-top2 差が5-15pp"
- 低確信: "top1-top2 差が5pp未満"

---

## 制約遵守

| 制約 | 結果 |
|------|------|
| UIは変更しない (ラベルのみ変更) | **OK** (metrics, カードは同じ構造) |
| draw_alert は独立軸として維持 | **OK** (変更なし) |
| 予測ロジックは変更しない | **OK** (hybrid_v9.1は変更なし) |
