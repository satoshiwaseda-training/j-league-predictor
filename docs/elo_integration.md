# ELO統合設計書 (elo_integration.md)

> 作成日: 2026-04-08

---

## 1. ELOの有効性 (実証済み)

### elo_only ベースライン結果 (val=2025, n=377)
- accuracy: 0.4828 (current v5 0.4642を上回る)
- F1 macro: 0.4056 (current v5 0.3410を大幅上回る)
- log_loss: 1.0394
- draw予測: 40件 (唯一drawを予測できたモデル)

### Optuna最適化での ELO 重み
- 現行シグモイドモデル: ELO重み = 25.6% (最大)
- 3ロジットモデル: ELO重み = 26.6% (最大)

→ **ELOは全モデルで最重要特徴量**

---

## 2. ELOパラメータ (最適化結果)

| パラメータ | 値 | 意味 |
|-----------|------|------|
| K factor | 27-44 | 更新の大きさ (文献値 20-40 と整合) |
| initial | 1500 | 初期レーティング |
| home_bonus | 25-100 | ホームアドバンテージ (ELOポイント) |

home_bonusの最適値にはバラつきがある (25-100)。
3logitでは25.3、現行ではは85.4。
理由: 3logitにはbias_homeが別途あるため、ELOのhome_bonusは小さくてよい。

---

## 3. 本番統合の設計

### 3.1 predict_logic.py への追加

```python
# EloSystem クラスを predict_logic.py に追加
# backtest_runner.py のものと同一実装
class EloSystem:
    def __init__(self, k=32.0, initial=1500.0, home_bonus=50.0): ...
    def get(self, team): ...
    def update(self, home, away, winner): ...
    def score_pair(self, home, away) -> tuple[float, float]: ...
```

### 3.2 3クラスへの変換

ELO差分は `score_pair()` で 0-1 のhome期待勝率に変換される。
3logit方式では:
- ELO差分 → raw_advantage の一部として他の特徴量と加重合算
- 接近度 (closeness) の算出にも寄与

### 3.3 Geminiプロンプトへの追加

```
## ELO レーティング
{home_team}: {home_elo:.0f} (リーグ平均1500)
{away_team}: {away_elo:.0f}
ELO差: {home_elo - away_elo:+.0f} (正=ホーム有利)
```

---

## 4. 注意事項

- シーズン跨ぎのELO引き継ぎが必要 (昇格組は初期値1500)
- ELOだけに過度に依存しないこと (現状26%で十分)
- 本番では過去試合データからELOを構築する必要がある
  → get_past_results() + get_historical_results() でデータ供給
