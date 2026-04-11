# 中確信が0件になる原因調査レポート

> 調査日: 2026-04-11

---

## 結論

**中確信(medium)カテゴリは構造的に0件になっていません。しかし特定条件下で極端に少なくなります。**

---

## 調査結果

### 1. 分類ロジックの正当性確認

`_classify_prediction` (app.py:980):
```python
h = int(pred.get("home_win_prob", 40))
d = int(pred.get("draw_prob", 25))
a = int(pred.get("away_win_prob", 35))
mx = max(h, d, a)
if mx >= 50:   confidence_level = "high"
elif mx >= 40: confidence_level = "medium"
else:          confidence_level = "low"
draw_alert = d >= 25 and closeness >= 0.5
```

**confidence_level と draw_alert は独立したフラグ。排他的ではない。**

### 2. サマリーカウント集計

```python
n_high = sum(1 for p in valid if p["classification"]["confidence_level"] == "high")
n_mid  = sum(1 for p in valid if p["classification"]["confidence_level"] == "medium")
n_low  = sum(1 for p in valid if p["classification"]["confidence_level"] == "low")
n_draw = sum(1 for p in valid if p["classification"]["draw_alert"])
```

**n_high + n_mid + n_low = n_valid** (排他的)
**n_draw は独立** (high/medium/low のいずれかと重複カウント)

→ 中確信が draw警戒に「吸われる」ことはない。両方にカウントされる。

### 3. hybrid_v9.1 出力分布 (2026全41試合)

| 確信度 | 件数 | 割合 |
|--------|------|------|
| high (>=50%) | 9 | 22.0% |
| **medium (40-50%)** | **12** | **29.3%** |
| low (<40%) | 20 | 48.8% |

| max_prob帯 | 件数 |
|-----------|------|
| <35 | 2 |
| 35-40 | 18 |
| **40-45** | **6** |
| **45-50** | **6** |
| 50-55 | 4 |
| 55-60 | 2 |
| 60+ | 3 |

**hybrid_v9.1 は medium を 12件出しており、「0件」ではない。**

### 4. クロス集計

| 組み合わせ | 件数 |
|------------|------|
| high + draw_alert | 4 |
| **medium + draw_alert** | **4** |
| low + draw_alert | 18 |
| total draw_alert | 26 |

**medium のうち 4件は draw警戒と重複、8件は draw警戒なし。**

### 5. モデル別の分布 (既存predictions.json 65件)

| モデル | high | medium | low |
|--------|------|--------|-----|
| statistical (v5旧) | 16% | **70%** | 14% |
| **gemini (v5旧)** | 11% | 46% | **43%** |
| **hybrid_v9.1** | 22% | **29%** | 49% |

**Gemini は確率を極端に振る傾向** があり、low に偏りやすい。

---

## 想定される原因

ユーザーが「中確信0件」と報告した時、以下のいずれか:

### A. Streamlit Cloudのキャッシュ問題で hybrid_v9.1 が動いていない
前回の修正前、`PRIMARY_MODEL_VERSION` のImportErrorで hybrid が使えず、
v7 + Gemini fallback になっていた。Geminiは low=43% に偏るため、
医ウィンドウの試合数が少なければ medium=0 もあり得る。

### B. 直近ボタン押下時の対象試合数が少なく、偶然 medium=0
もし予測対象試合が 4-5件しかなければ、偶然0件になる可能性。

### C. 表示は正しいが、hybrid_v9.1 が実際に low寄り出力
2026で hybrid_v9.1 の 48.8% が low → 対象試合数が少ない場合は medium=0 あり得る。

---

## 検証方法

ワンボタン予測実行後、実際にカードに表示された試合の **max_prob** を確認:

### 期待される分布 (hybrid_v9.1 での2026試合)
- 高: 約 22%
- 中: 約 29%
- 低: 約 49%
- draw警戒: 約 63%

### 実際が異なる場合
- **中=0%**: Gemini fallback中 (PRIMARY_MODEL_VERSIONがimportできていない)
- **高=0%, 中=0%, 低=100%**: 低確信寄りモデル
- **高=80%+**: 高確信寄り (Geminiの極端振り)

---

## 改善案

### 案1: 中確信と draw警戒の併記表示
サマリーに「中確信 (うち draw警戒N件)」を追加表示:
```
高確信 5試合
中確信 12試合 (うち draw警戒 4件)
低確信 20試合
Draw警戒 26試合
```

### 案2: 確信度閾値の微調整
現状: high>=50 / medium 40-50 / low<40
候補: high>=48 / medium 38-48 / low<38
→ medium の範囲を広げることで該当件数を増やす

### 案3: サマリー文言の補足
「中確信 0試合」の横に注釈:
```
中確信 0試合 (high/lowに集中、多くがdraw警戒)
```

### 案4: 分布プレビュー表示
サマリー直下にヒストグラム:
```
max_prob: ▁▃▇▅▃▂▁
       30 40 50 60 70
```

---

## 推奨対応

1. **まず Streamlit Cloud の再デプロイ確認**
   - キャッシュが更新されれば hybrid_v9.1 が動き medium=12 (29%) に戻る
   
2. **サマリー集計に draw警戒の内訳を追加** (案1)
   - 中確信と draw警戒の重複可視化で誤解を防ぐ

3. **文言補足** (案3)
   - 中=0件時に「fallback中の可能性」を示唆
