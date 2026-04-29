"""
scripts/weekend_review_2026-04-25.py

2026-04-22〜04-26 の予測 vs 実結果の照合 + 外れ原因分析。
予測フォルダ: C:\\Users\\User\\OneDrive\\Desktop\\Jリーグ予測結果\\
実結果: data_fetcher.get_past_results + Jleague 公式 (4/26 J2 / 4/25 J1)
"""
from __future__ import annotations
import os, json, math, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PRED_DIR = Path("/sessions/magical-brave-heisenberg/mnt/Jリーグ予測結果")
DATES = ["2026-04-22", "2026-04-24", "2026-04-25", "2026-04-26"]

# ─── 予測の読み込み ───
predictions = []
for div in ["J1", "J2"]:
    for d in DATES:
        path = PRED_DIR / div / d
        if not path.exists():
            continue
        for jf in sorted(path.glob("*.json")):
            try:
                p = json.loads(jf.read_text(encoding="utf-8"))
                pr = p.get("prediction_result") or p.get("entry", {}).get("prediction") or {}
                m = p.get("match") or p.get("entry", {}).get("match") or {}
                predictions.append({
                    "div": div,
                    "date": m.get("date", d),
                    "home": m.get("home"),
                    "away": m.get("away"),
                    "h_prob": pr.get("home_win_prob"),
                    "d_prob": pr.get("draw_prob"),
                    "a_prob": pr.get("away_win_prob"),
                    "pred_winner": pr.get("pred_winner"),
                    "predicted_score": pr.get("predicted_score"),
                    "confidence": pr.get("confidence"),
                    "model": pr.get("model"),
                    "hybrid_selection": pr.get("hybrid_selection"),
                })
            except Exception as e:
                print(f"failed: {jf.name}: {e}", file=sys.stderr)

# ─── 実結果 (Jリーグ公式から取得した先週末分) ───
ACTUALS = {
    # J1 4/22  → 公式リーグ戦ではない (ACL or 別大会)。除外
    # J1 4/24
    ("J1", "2026-04-24", "FC東京",       "水戸ホーリーホック"):     {"home":5, "away":2, "winner":"home", "score":"5-2"},
    ("J1", "2026-04-24", "柏レイソル",     "鹿島アントラーズ"):       {"home":0, "away":1, "winner":"away", "score":"0-1"},
    # J1 4/25
    ("J1", "2026-04-25", "浦和レッズ",     "横浜F・マリノス"):        {"home":2, "away":3, "winner":"away", "score":"2-3"},
    ("J1", "2026-04-25", "川崎フロンターレ",  "ジェフユナイテッド千葉"):    {"home":2, "away":1, "winner":"home", "score":"2-1"},
    ("J1", "2026-04-25", "清水エスパルス",   "名古屋グランパス"):       {"home":0, "away":2, "winner":"away", "score":"0-2"},
    ("J1", "2026-04-25", "ファジアーノ岡山",  "アビスパ福岡"):         {"home":2, "away":0, "winner":"home", "score":"2-0"},
    ("J1", "2026-04-25", "サンフレッチェ広島", "セレッソ大阪"):         {"home":2, "away":1, "winner":"home", "score":"2-1"},
    ("J1", "2026-04-25", "V・ファーレン長崎", "ガンバ大阪"):          {"home":1, "away":1, "winner":"draw", "score":"1-1 (5PK6)"},
    # J2 4/25
    ("J2", "2026-04-25", "FC琉球",       "大分トリニータ"):         {"home":0, "away":0, "winner":"draw", "score":"0-0"},
    # J2 4/26
    ("J2", "2026-04-26", "ヴァンラーレ八戸",  "ブラウブリッツ秋田"):     {"home":1, "away":0, "winner":"home", "score":"1-0"},
    ("J2", "2026-04-26", "松本山雅FC",     "AC長野パルセイロ"):      {"home":0, "away":1, "winner":"away", "score":"0-1"},
    ("J2", "2026-04-26", "高知ユナイテッドSC", "徳島ヴォルティス"):       {"home":2, "away":1, "winner":"home", "score":"2-1"},
    ("J2", "2026-04-26", "愛媛FC",       "カマタマーレ讃岐"):        {"home":3, "away":0, "winner":"home", "score":"3-0"},
}

# ─── 結合 ───
rows = []
for p in predictions:
    key = (p["div"], p["date"], p["home"], p["away"])
    actual = ACTUALS.get(key)
    rows.append({**p, "actual": actual})

print("=" * 120)
print(f"  予測 vs 実結果 (n={len(rows)})")
print("=" * 120)
print(f"{'div':<3} {'date':<11} {'カード':<32} {'pred%':<13} {'pred_score':<11} {'pred_W':<5} "
      f"{'actual':<14} {'判定':<5}")
print("-" * 120)

correct_count = 0
total_with_actual = 0
miss_details = []
for r in rows:
    card = f"{r['home']:<14}vs{r['away']:<14}"[:32]
    pct = f"{r['h_prob']}/{r['d_prob']}/{r['a_prob']}"
    a = r["actual"]
    if a is None:
        a_text, judge = "結果なし", "—"
    else:
        a_text = f"{a['score']} ({a['winner'][:4]})"
        is_correct = r["pred_winner"] == a["winner"]
        judge = "○" if is_correct else "✗"
        total_with_actual += 1
        if is_correct: correct_count += 1
        else: miss_details.append({**r, "actual": a})
    print(f"{r['div']:<3} {r['date']:<11} {card:<32} {pct:<13} {r['predicted_score'] or '-':<11} {r['pred_winner'] or '-':<5} {a_text:<14} {judge:<5}")

print()
print(f"全体正答率: {correct_count}/{total_with_actual} = {(correct_count/total_with_actual*100) if total_with_actual else 0:.1f}%")

# ─── ラベル別集計 ───
# 戦略ラベルは confidence と (確率) から逆算する
# （現在の保存データには label が入っていないので diff から推定）
def calc_diff(r):
    probs = sorted([r['h_prob'] or 0, r['d_prob'] or 0, r['a_prob'] or 0], reverse=True)
    return probs[0] - probs[1]

def calc_label(r):
    """簡易: confidence + diff + probs から戦略ラベルを推定"""
    if r['confidence'] is None:
        return "?"
    h, d, a = r['h_prob'] or 0, r['d_prob'] or 0, r['a_prob'] or 0
    diff = calc_diff(r)
    closeness = max(0.0, 1.0 - abs(h - a) / 50.0)  # 簡易closeness
    draw_alert = d >= 25 and closeness >= 0.5
    mx = max(h, d, a)
    if r['confidence'] == 'high':
        if draw_alert and mx >= 60: return "最強"
        if draw_alert and mx < 60:  return "組み合わせ (改修B降格)"
        return "本命"
    if r['confidence'] == 'medium':
        if draw_alert: return "波乱狙い"
        return "組み合わせ"
    if r['confidence'] == 'low':
        if draw_alert: return "スキップ"
        return "見送り"
    return "?"

print()
print("=" * 120)
print("  ラベル別 集計")
print("=" * 120)
from collections import defaultdict
by_label = defaultdict(lambda: {"n":0, "hit":0, "matches":[]})
for r in rows:
    if r["actual"] is None: continue
    lbl = calc_label(r)
    by_label[lbl]["n"] += 1
    if r["pred_winner"] == r["actual"]["winner"]:
        by_label[lbl]["hit"] += 1
    by_label[lbl]["matches"].append(r)

for lbl, v in sorted(by_label.items(), key=lambda x: -x[1]["n"]):
    acc = v["hit"] / v["n"] * 100 if v["n"] else 0
    print(f"  {lbl:<22} n={v['n']:>3}  hit={v['hit']:>3}  acc={acc:>5.1f}%")

# ─── 外れ試合の詳細 ───
print()
print("=" * 120)
print(f"  外れ試合の詳細 (n={len(miss_details)})")
print("=" * 120)
for r in miss_details:
    a = r["actual"]
    pct = f"H{r['h_prob']}% D{r['d_prob']}% A{r['a_prob']}%"
    diff = calc_diff(r)
    lbl = calc_label(r)
    print(f"  {r['div']} {r['date']} {r['home']} vs {r['away']}")
    print(f"    予測: {pct} → {r['pred_winner']} ({lbl}, conf={r['confidence']}, diff={diff})")
    print(f"    実結果: {a['score']} ({a['winner']})")
    print()

# 保存
out = Path("/sessions/magical-brave-heisenberg/mnt/j-league-predictor/backtest_results/weekend_review_2026-04-25.json")
out.parent.mkdir(exist_ok=True, parents=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump({
        "weekend": "2026-04-22 to 2026-04-26",
        "n_total": total_with_actual,
        "n_correct": correct_count,
        "accuracy": correct_count / total_with_actual if total_with_actual else None,
        "by_label": {k: {kk: vv for kk, vv in v.items() if kk != "matches"} for k, v in by_label.items()},
        "rows": rows,
        "misses": [{**m, "label": calc_label(m), "diff": calc_diff(m)} for m in miss_details],
    }, f, ensure_ascii=False, indent=2, default=str)
print(f"saved: {out}")
