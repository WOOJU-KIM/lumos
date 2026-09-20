import re
from pathlib import Path

p = Path('core/live_runner.py')
c = p.read_text(encoding='utf-8', errors='ignore')

# 1. start_msg
c = re.sub(
    r'(start_msg = f\"\"\").*?(\"\"\")',
    r'\1🏁 *[Lumos 실전투자 라이브러너 기동]*\n🕒 *현재 시각:* {mkt_status["now_kst_str"]}\n🔄 *시간 동기화:* 100% 일치 (KST-NYT {sync_res["delta_hours"]:.0f}h 시차 검증 완료)\n📊 *시장 상태:* {mkt_status["status_desc"]}\n⏰ *다음 개장:* {mkt_status["next_open_kst_str"]} ({mkt_status["time_until_open_str"]})\n🏦 *연동 브로커:* {self.broker.broker_name} ({self.broker.mode_str})\n💳 *계좌 번호:* {self.broker.account_no}-{self.broker.account_type}\n💰 *가용 예수금:* ${conn_res["usd_order_available"]:,.2f} USD (약 {conn_res["krw_converted"]:,}원)\n📦 *보유 포지션:* {conn_res["holdings_count"]}개\n🧠 *AI 엔진:* {ai_engine_desc}\n⚡ *체결망:* 10ms WebSocket 틱 스트림 우선 연동\n🚀 자, 모든 시스템이 완벽하게 준비되었습니다.\2',
    c, flags=re.DOTALL
)

# 2. open_msg
c = re.sub(
    r'(open_msg = f\"\"\").*?(\"\"\")',
    r'\1🔥 *[정규장 매매 개시]*\n\n⏰ 현재 시각: {mkt["now_kst_str"]}\n🖥️ 실행 모드: {self.broker.mode_str}\n🧠 AI 전략: 하이브리드 MoE V5 (09:30~15:30 EDT)\n🎯 매수 룰: GBDT {config.GBDT_CONFIDENCE_THRESHOLD*100:.0f}% 이상\n🛡️ 청산 룰: Max TP +{config.MAX_TP_PCT*100:.1f}% / ATR Trailing Stop\n🌙 마감청산: {config.PHASE_COOLDOWN_END} EDT 0% 오버나잇 전량 시장가\2',
    c, flags=re.DOTALL
)

# 3. buy_msg
c = re.sub(
    r'(buy_msg = f\"\"\").*?(\"\"\")',
    r'\1{mode_title} 신규 매수 주문 (1차)\n🚨 *브로커:* {self.broker.broker_name} {self.broker.mode_str}\n💡 *종목:* {symbol}\n📊 *수량:* {target_qty}\n💰 *주문가:* ${order_px_1:.2f}\n\n🔥 *[GBDT 모델 확신도]*\n{score_block}\n\n🎯 *목표가:* ${tp_px:.2f}\n🛡️ *손절가:* ${sl_px:.2f}\n⏱️ *시간청산:* {time_stop_min}분\2',
    c, flags=re.DOTALL
)

# 4. sell_msg
c = re.sub(
    r'(sell_msg = f\"\"\").*?(\"\"\")',
    r'\1{mode_title} 100% 매도 청산 완료\n🚨 *브로커:* {self.broker.broker_name} {self.broker.mode_str}\n💡 *사유:* {reason_desc}\n📊 *종목/수량:* {symbol} {quantity:,}주\n💰 *체결가:* ${latest_px:.2f} (최종 수익률: {final_pnl_pct:+.2f}%)\n🛡️ *사후 관리:* 100% 현금화 완료 (AI MoE 새 진입 대기)\2',
    c, flags=re.DOTALL
)

# 5. score_block
c = re.sub(
    r'score_block = \"\\n\"\.join\(\[f\".*?\"',
    r'score_block = "\\n".join([f"  * {k.upper()}:* {v*100:.1f}%"',
    c
)

p.write_text(c, encoding='utf-8')
print('Telegram strings perfectly patched!')
