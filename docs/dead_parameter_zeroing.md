# Dead Parameter 0化レポート (dead_parameter_zeroing.md)

> 作成日: 2026-04-09

---

## 結果: 0化の影響はゼロ

| 指標 | BEFORE | AFTER | 差分 |
|------|--------|-------|------|
| accuracy | 0.4695 | 0.4695 | 0.0000 |
| F1 macro | 0.4222 | 0.4222 | 0.0000 |
| log loss | 1.0663 | 1.0663 | 0.0000 |
| Brier | 0.6440 | 0.6440 | 0.0000 |
| draw# | 59 | 59 | 0 |

## 理由

死にパラメータ6個は:
- データ供給なし → 常に home_score=away_score=0.5
- 差分=0 → contribution=0
- raw_home_advantage への寄与は元から0
- closeness は raw_adv から計算 → 影響なし

重みの再正規化も無影響:
- backtest_runner.py の predict_current_model 内で `if tw > 0 and abs(tw-1.0) > 0.01: raw_adv /= tw`
- predict_logic.py の calculate_parameter_contributions では正規化なし
  しかし差分が0のパラメータは加算しても0

## 結論

0化は「コードの衛生管理」としては有意義だが、数値的影響はゼロ。
確率品質改善の鍵は死にパラメータではなく、**有効パラメータの重み配分とロジットスケール**にある。
