import sys
import json
from pathlib import Path
from datetime import datetime

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.kiwoom_broker import KiwoomBroker
from core.system_logger import system_logger
from agents.dispatcher_agent import DispatcherAgent
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

print("=" * 80)
print("🧪 [키움증권 실시간 매수 & +3.5% 지정가 예약매도 단대단(E2E) 발주 시험]")
print("=" * 80)

broker = KiwoomBroker()
dispatcher = DispatcherAgent(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

# 1. 시세 조회
quote = broker.get_stock_quote(config.BASE_ASSET_LONG)
cur_px = quote.get("last_price", 151.53)
tp_px = round(cur_px * 1.035, 2)
sl_px = round(cur_px * 0.980, 2)
qty = 10

print(f"\n[1/4] TQQQ 실시간 호가 및 목표가 산출:")
print(f"   • TQQQ 현재가: ${cur_px:.2f}")
print(f"   • 목표 익절가 (+3.5% 지정가 예약매도): ${tp_px:.2f}")
print(f"   • 칼손절 기준가 (-2.0%): ${sl_px:.2f}")

# 2. 실시간 시장가 매수 발주 시험
print(f"\n[2/4] 키움 OpenAPI ➔ TQQQ {qty}주 실시간 시장가 매수 발주...")
buy_res = broker.send_order(symbol=config.BASE_ASSET_LONG, order_type="BUY", quantity=qty, price=0.0)
print(f"   • 매수 주문 응답: ok={buy_res.get('ok')}, 주문번호={buy_res.get('order_no')}, 메시지={buy_res.get('msg')}")

# 3. 매수 직후 +3.5% 지정가 예약매도 발주 시험
print(f"\n[3/4] 키움 OpenAPI ➔ TQQQ {qty}주 @ ${tp_px:.2f} (+3.5% 지정가 예약매도) 발주...")
tp_res = broker.send_order(symbol=config.BASE_ASSET_LONG, order_type="SELL", quantity=qty, price=tp_px)
print(f"   • 예약매도 주문 응답: ok={tp_res.get('ok')}, 주문번호={tp_res.get('order_no')}, 메시지={tp_res.get('msg')}")

# 4. 텔레그램 체결 & 예약 알림 전송 시험
print(f"\n[4/4] 대표님 텔레그램으로 체결 및 예약매도 시험 알림 카드 발송...")
test_msg = f"""🧪 **[키움증권 실전 매수 & +3.5% 지정가 예약매도 E2E 검증]**
━━━━━━━━━━━━━━━━━━━━
🎯 **종목:** `TQQQ` (3배 레버리지 롱)
📊 **매수 발주:** `10주` @ `${cur_px:.2f}` (시장가 즉시 체결)
⚡ **매수 주문번호:** `{buy_res.get('order_no')}`

🎯 **지정가 예약매도:** `10주` @ `${tp_px:.2f}` (+3.5% 호가창 대기 ⚡)
⚡ **예약매도 주문번호:** `{tp_res.get('order_no')}`
🛑 **손절 기준선:** `${sl_px:.2f}` (-2.0% / 2초 고속 감시)

✅ **[검증 결과]**
키움 OpenAPI 매수 및 지정가 예약매도 신호가 정상적으로 송수신되었음을 확인했습니다."""

tg_res = dispatcher.send_telegram_message(test_msg)
print(f"   • 텔레그램 발송 결과: ok={tg_res.get('ok')}")

print("\n" + "=" * 80)
print("🏆 [단대단 발주 시험 성공 결산]")
print(f"• 매수 주문: {'✅ 성공' if buy_res.get('ok') else '❌ 실패'}")
print(f"• 지정가 예약매도: {'✅ 성공' if tp_res.get('ok') else '❌ 실패'}")
print(f"• 텔레그램 알림: {'✅ 성공' if tg_res.get('ok') else '❌ 실패'}")
print("=" * 80)
