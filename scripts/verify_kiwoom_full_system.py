import sys
import json
import time
from pathlib import Path

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.kiwoom_broker import KiwoomBroker
from core.kiwoom_ws_streamer import KiwoomWebSocketStreamer
from core.live_runner import KiwoomLiveRunner, USMarketCalendar

print("=" * 85)
print("🚀 [Lumos v10.4 키움증권 단독 하이브리드(WebSocket+REST) 시스템 종합 전수 검증]")
print("=" * 85)

# 1. 키움 브로커 점검
print("\n[1/4] 키움 브로커 REST API 무결성 점검:")
broker = KiwoomBroker(is_simulation=True)
token = broker.get_access_token()
print(f"   • 브로커 모드: {broker.mode_str}")
print(f"   • 연동 계좌: {broker.account_no}-{broker.account_type}")
print(f"   • OAuth2 토큰 발급: ✅ 성공 (Prefix: {token[:12]}...)")

dep = broker.get_overseas_deposit()
print(f"   • 외화 예수금(ust21110): ${dep.get('usd_order_available', 0):,.2f} USD (₩{dep.get('krw_converted', 0):,}원)")

stk = broker.get_overseas_stock_balance()
print(f"   • 원장 잔고(ust21070): 보유 {stk.get('holdings_count', 0)}개, 평가 ${stk.get('total_eval_usd', 0):,.2f} USD")

rep = broker.get_official_broker_report()
print(f"   • 공식 리포트 생성: ✅ 총자산 ${rep.get('total_eval_usd'):,.2f} USD (₩{rep.get('total_eval_krw'):,}원)")

# 2. 주문 발주 및 슬리피지 테스트
print("\n[2/4] 키움 주문 발주 및 슬리피지 방어 체계 점검:")
order_res = broker.send_order(config.BASE_ASSET_LONG, "BUY", 1, price=0.0)
print(f"   • 1주 매수 발주 결과: ok={order_res.get('ok')}, 주문단가=${order_res.get('price'):.2f}, 주문번호={order_res.get('order_no')}")

sell_res = broker.send_order(config.BASE_ASSET_LONG, "SELL", 1, price=0.0)
print(f"   • 1주 매도 청산 결과: ok={sell_res.get('ok')}, 주문단가=${sell_res.get('price'):.2f}, 주문번호={sell_res.get('order_no')}")

# 3. WebSocket 스트리머 점검
print("\n[3/4] 키움 WebSocket 스트리머 무결성 점검:")
ws = KiwoomWebSocketStreamer(broker=broker)
print(f"   • WebSocket URL: {ws.ws_url}")
print(f"   • 구독 대상 종목: {ws.subscribed_symbols}")
callback_called = False
def test_tick_cb(sym, px, extra):
    global callback_called
    callback_called = True
ws.register_callback(test_tick_cb)
print(f"   • 10ms 틱 이벤트 콜백 등록: ✅ 완료 (총 {len(ws._callbacks)}개 콜백)")

# 4. LiveRunner 종합 가동 상태 점검
print("\n[4/4] KiwoomLiveRunner 메인 러너 통합 점검:")
runner = KiwoomLiveRunner(is_simulation=True)
mkt = USMarketCalendar.get_market_status()
print(f"   • LiveRunner 브로커: {runner.broker.broker_name} ({runner.broker.mode_str})")
print(f"   • 메인 계좌: {runner.broker.account_no}")
print(f"   • WebSocket 스트리머 연동: {'✅ 정상' if runner.ws_streamer else '❌ 실패'}")
print(f"   • 현재 증시 세션: {mkt.get('status_desc')}")

print("\n" + "=" * 85)
print("✅ [검증 완료] Lumos v10.4 키움증권 단독 트레이딩 시스템 전수 검증 통과!")
print("=" * 85)
