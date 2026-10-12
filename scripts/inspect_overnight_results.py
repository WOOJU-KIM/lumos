import sys
import json
import sqlite3
from datetime import datetime
from pathlib import Path

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.hybrid_broker import HybridUniversalBroker
from core.live_runner import USMarketCalendar
from core.system_logger import system_logger
from core.data_lake import MarketDataLake
from core.moe_orchestrator import MoEMetaOrchestrator

print("=" * 80)
print("🔍 [Lumos 새벽 정규장 모의 거래 및 AI 의사결정 전수 점검]")
print("=" * 80)

# 1. 증권사 OpenAPI 직접 실시간 통신 조회
broker = HybridUniversalBroker(is_simulation=True)
rep = broker.get_official_broker_report()

print("\n1. 🏛 [증권사 직접 조회 원장 공식 데이터]")
print(f"• 데이터 출처: {rep.get('source')}")
print(f"• 연동 계좌: {rep.get('account_no')}")
print(f"• 주문가능 외화: ${rep.get('avail_usd', 0.0):,.2f} USD")
print(f"• 총 평가 자산: ${rep.get('total_eval_usd', 0.0):,.2f} USD (₩{rep.get('total_eval_krw', 0):,}원)")
print(f"• 환율: {rep.get('exchange_rate', 1411.0):,.2f} KRW/USD")
print(f"• 보유 주식 수: {rep.get('holdings_count', 0)}개 (오버나잇 0% 현금화 완료)")
print(f"• 공식 실현손익: ${rep.get('realized_pnl_usd', 0.0):,.2f} USD ({rep.get('realized_rate_pct', 0.0):+.2f}%)")

execs = rep.get('executions', [])
print(f"• 증권사 체결 건수: {len(execs)}건")
for ex in execs:
    print(f"  - {ex}")

# 2. 새벽 장세 분석 및 AI 능동 의사결정 (TQQQ vs SQQQ)
print("\n2. 🧠 [새벽 장세 국면 및 AI 능동 SQQQ/TQQQ 스캔 분석]")
lake = MarketDataLake()
long_15m = lake.load_candles(config.BASE_ASSET_LONG, "15m")
short_15m = lake.load_candles(config.BASE_ASSET_SHORT, "15m")

moe = MoEMetaOrchestrator(confidence_threshold=0.75)
long_eval = moe.evaluate_dual_filter_signal(long_15m, threshold=0.75)
short_eval = moe.evaluate_dual_filter_signal(short_15m, threshold=0.75)

print(f"• 현재 MoE 게이팅 스캔 결과:")
print(f"  - TQQQ 스캔: 지목 [{long_eval.get('expert_desc')}] | 확신도: {long_eval.get('gating_confidence', 0)*100:.1f}점 | 방향: {long_eval.get('direction')} | 승인: {long_eval.get('is_approved')}")
print(f"  - SQQQ 스캔: 지목 [{short_eval.get('expert_desc')}] | 확신도: {short_eval.get('gating_confidence', 0)*100:.1f}점 | 방향: {short_eval.get('direction')} | 승인: {short_eval.get('is_approved')}")

# 3. 섀도우 원장 DB 조회
print("\n3. 💾 [DB 적재 의사결정 및 체결 내역]")
conn = sqlite3.connect("data/shadow_trades.db")
conn.row_factory = sqlite3.Row
cur = conn.cursor()
rows = cur.execute("SELECT * FROM shadow_trades ORDER BY rowid DESC LIMIT 5;").fetchall()
for r in rows:
    print(f"• [{r['trade_date']} {r['entry_time']}] {r['ticker']} | {r['direction']} | 지목: {r['selected_expert']} | 확신도: {r['gating_weight']*100:.1f}점 | PnL: {r['pnl_pct']}% | 사유: {r['exit_reason']}")

print("=" * 80)
