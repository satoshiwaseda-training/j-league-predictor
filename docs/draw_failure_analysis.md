# draw予測構造欠陥分析 (draw_failure_analysis.md)

> 作成日: 2026-04-08

---

## 1. 結論

**drawは現行ロジックにおいて構造的にargmax予測されることが不可能である。**

`advantage_to_probs()` および `predict_current_model()` において、
raw_advantage が -1.0 ~ +1.0 の **全範囲** で draw 確率が home/away 確率を超えることはない。
これはパラメータ調整では修正できない **設計上の欠陥** である。

---

## 2. 根本原因: draw = 残差

### 現行の確率変換式

```python
shift = tanh(raw_advantage * K) * M      # K=3, M=0.30
h = clamp(base_home + shift, 0.05, 0.90)  # base_home=0.40
a = clamp(base_away - shift, 0.05, 0.90)  # base_away=0.35
d = clamp(1.0 - h - a, 0.05, 0.35)        # ← draw は残差
```

### なぜ d < h かつ d < a が常に成立するか

**ケース1: raw_advantage = 0 (完全均衡)**
```
h = 0.40, a = 0.35, d = 1.0 - 0.40 - 0.35 = 0.25
argmax = home (0.40 > 0.35 > 0.25)
```
→ 完全均衡でもdrawは3位。

**ケース2: raw_advantage = -1.0 (最大アウェイ有利)**
```
shift = tanh(-3) * 0.30 = -0.2985
h = 0.40 - 0.2985 = 0.1015
a = 0.35 + 0.2985 = 0.6485
d = 1.0 - 0.1015 - 0.6485 = 0.25
argmax = away (0.6485 > 0.25 > 0.1015)
```
→ d = 0.25 で固定。hが下がってもaが上がるためdは常に~0.25。

**一般的証明:**
```
d = 1 - h - a
  = 1 - (base_home + shift) - (base_away - shift)
  = 1 - base_home - base_away
  = 1 - 0.40 - 0.35
  = 0.25 (定数!)
```

**drawは raw_advantage に依存しない定数 0.25 である。**
shift が h に加算されると同時に a から減算されるため、h+a は常に一定。
よって d = 1 - (h+a) も常に一定 = 0.25。

clamp の影響で極端な値では微変動するが、argmax でdrawが選ばれる領域は存在しない。

---

## 3. draw_closeness_boost が機能しない理由

Optuna最適化で `draw_closeness_boost=0.179` が選ばれたが、drawは依然0件。

```python
closeness = max(0, 1.0 - |raw_adv| * 2)  # 0~1
draw_boost = 0.179 * closeness            # 最大0.179
d = clamp((1.0 - h - a) + draw_boost, 0.05, 0.50)
```

raw_adv=0 のとき closeness=1.0、draw_boost=0.179:
```
d = 0.25 + 0.179 = 0.429
```

しかし正規化後:
```
h=0.40, a=0.35, d=0.429  → total=1.179
h_norm = 0.40/1.179 = 0.339
a_norm = 0.35/1.179 = 0.297
d_norm = 0.429/1.179 = 0.364
argmax = draw!
```

...これは理論上drawが選ばれるはず。しかし実際にはraw_adv=0の試合はほぼ存在しない。
多くの試合でraw_adv != 0 のため closeness < 1 となり、boostが不十分になる。

**真の問題**: draw_closeness_boostは理論的にはdrawを出せるが、
実際のデータで raw_adv が十分0に近い試合が少なく、閾値を超えない。

---

## 4. 設計方式の比較

### 方式A: 3クラス softmax

```python
logit_home = w_h * features + bias_h
logit_draw = w_d * features + bias_d
logit_away = w_a * features + bias_a
probs = softmax([logit_home, logit_draw, logit_away])
```

| 観点 | 評価 |
|------|------|
| 実装容易性 | 中 (特徴量→3ロジットの変換が必要) |
| 既存コード影響 | 中 (advantage_to_probs の置換) |
| 解釈性 | 高 (各クラスに明示的なスコア) |
| draw再現性 | **高** (drawに独自ロジットがある) |
| calibration | 高 (softmaxは確率的に一貫) |

### 方式B: 3ロジット方式 (home/away差分 + draw独立)

```python
strength_diff = home_score - away_score   # 勝敗方向
draw_signal = closeness_measure            # draw傾向
logit_h = alpha * strength_diff + beta
logit_d = gamma * draw_signal + delta
logit_a = -alpha * strength_diff + epsilon
```

| 観点 | 評価 |
|------|------|
| 実装容易性 | **高** (既存raw_advantageを流用) |
| 既存コード影響 | **低** (変換関数のみ置換) |
| 解釈性 | **高** (差分→勝敗、接近度→draw) |
| draw再現性 | **高** |
| calibration | 中 (パラメータ設計に依存) |

### 方式C: ELO差分ベース ordered/multinomial 近似

```python
elo_diff = elo_home - elo_away + home_bonus
p_home = sigmoid(elo_diff / scale - draw_width)
p_away = sigmoid(-elo_diff / scale - draw_width)
p_draw = 1 - p_home - p_away
```

| 観点 | 評価 |
|------|------|
| 実装容易性 | 中 |
| 既存コード影響 | 高 (ELO依存) |
| 解釈性 | 高 (サッカー分析で標準的) |
| draw再現性 | **高** |
| calibration | **高** (理論的に確立) |

---

## 5. 推奨: 方式B (3ロジット) を第一選択

理由:
1. 既存の `raw_advantage` をそのまま home/away 差分として使える
2. drawに独立したスコア (closeness信号) を持たせる
3. softmaxで3クラス確率に変換
4. 最小変更で構造問題を解決
5. 将来的にパラメータ学習も容易

実装イメージ:
```python
def advantage_to_probs_v2(raw_adv, closeness, draw_bias=0.0):
    logit_h = raw_adv * scale_ha + bias_h
    logit_a = -raw_adv * scale_ha + bias_a
    logit_d = closeness * scale_d + draw_bias
    return softmax([logit_h, logit_d, logit_a])
```
