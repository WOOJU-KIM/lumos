import sys
import json
import sqlite3
from pathlib import Path

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 80)
print("🔍 [TQQQ 783주 매수 체결 원인 정밀 포렌식 분석]")
print("=" * 80)

# 1. system_logs.jsonl 검색
logs_file = PROJECT_ROOT / "data" / "system_logs.jsonl"
if logs_file.exists():
    with open(logs_file, "r", encoding="utf-8") as f:
        lines = [json.loads(l.strip()) for l in f if l.strip()]
    
    print(f"• 전체 시스템 로그 수: {len(lines):,}건")
    print("• 매수/주문/TQQQ 관련 로그 목록:")
    for l in lines:
        msg = l.get("message", "")
        src = l.get("source", "")
        lvl = l.get("level", "")
        dt = l.get("datetime", "")
        if any(w in msg.lower() for w in ["tqqq", "매수", "autoexecution", "783", "124.5", "buy"]):
            print(f"  [{dt}][{lvl}][{src}] {msg}")

# 2. shadow_trades.db 검색
conn = sqlite3.connect("data/shadow_trades.db")
conn.row_factory = sqlite3.Row
cur = conn.cursor()
rows = cur.execute("SELECT * FROM shadow_trades WHERE ticker = config.BASE_ASSET_LONG ORDER BY rowid DESC LIMIT 10;").fetchall()
print(f"\n• Shadow Trades TQQQ 최근 기록 ({len(rows)}건):")
for r in rows:
    print(f"  - [{r['trade_date']} {r['entry_time']}] {r['ticker']} {r['entry_price']}$ ➔ 지목: {r['selected_expert']} (확신도: {r['gating_weight']*100:.1f}점) | 이유: {r['exit_reason']}")

print("=" * 80)
