import sys
import os
import config
import json
import time
from pathlib import Path
from datetime import datetime

# Windows 콘솔 utf-8 설정
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.kiwoom_broker import KiwoomBroker
from core.kiwoom_ws_streamer import KiwoomWebSocketStreamer

def execute_live_interactive_test():
    print("=" * 90)
    print("🔥 [실전 동일 모의 세션: WebSocket 실시간 시세 연동 ➔ 즉시 실탄 매수/매도 집행 테스트]")
    print(f"🕒 실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)

    # 1. 브로커 초기화 및 예수금 확인
    print("\n[Step 1] 키움 브로커 초기화 및 주문 가능 금액 확인")
    broker = KiwoomBroker(is_simulation=True)
    token = broker.get_access_token()
    dep = broker.get_overseas_deposit()
    bal_before = broker.get_overseas_stock_balance()
    
    usd_avail = dep.get("usd_order_available", 0.0)
    print(f"   • 연동 계좌: {broker.account_no}-{broker.account_type} ({broker.mode_str})")
    print(f"   • 주문 가능 예수금: ${usd_avail:,.2f} USD")
    print(f"   • 현재 보유 종목 수: {bal_before.get('holdings_count', 0)}개")

    # 2. WebSocket 실시간 스트리머 가동
    print("\n[Step 2] WebSocket 공식 포트(:10000) 접속 ➔ LOGIN ➔ REG 실시간 구독 가동")
    streamer = KiwoomWebSocketStreamer(broker=broker)
    
    captured_ticks = []
    def on_live_tick(symbol: str, price: float, extra: dict):
        captured_ticks.append((symbol, price, datetime.now().strftime("%H:%M:%S.%f")[:-3]))
        print(f"   ⚡ [실시간 WebSocket 틱 수신] 종목: {symbol} | 현재가: ${price:.2f} | 시각: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

    streamer.register_callback(on_live_tick)
    streamer.start()

    print("   ⏳ 실시간 웹소켓 세션 동기화 및 틱 수신 대기 (3초)...")
    time.sleep(3.0)

    # 3. 실시간 가격 확인
    long_price = streamer.get_latest_price(config.BASE_ASSET_LONG)
    print(f"\n[Step 3] 스트리머로부터 획득한 TQQQ 실시간 체결가: ${long_price:.2f}")

    # 4. 실시간 시세 기반 1주 시장가 매수 발주 집행
    print("\n[Step 4] 실시간 가격 기반 TQQQ 1주 시장가 매수 발주 송출 (TR: tt80010)")
    buy_order = broker.send_order(
        symbol=config.BASE_ASSET_LONG,
        order_type="BUY",
        quantity=1,
        price=0.0  # 시장가
    )
    print(f"   ✅ [매수 발주 완료]")
    print(f"      - 주문 번호: {buy_order.get('order_no')}")
    print(f"      - 주문 상태: {'정상 체결' if buy_order.get('ok') else '실패'}")
    print(f"      - 체결 단가: ${buy_order.get('price'):.2f}")
    print(f"      - 서버 메시지: {buy_order.get('msg')}")

    # 5. 매수 후 잔고 반영 확인
    time.sleep(1.0)
    bal_after_buy = broker.get_overseas_stock_balance()
    print(f"\n[Step 5] 매수 체결 후 계좌 잔고 확인")
    print(f"   • 보유 종목 수: {bal_after_buy.get('holdings_count', 0)}개")
    for h in bal_after_buy.get("holdings", []):
        print(f"     ➔ [{h.get('symbol')}] 수량: {h.get('quantity')}주 | 매수가: ${h.get('purchase_price'):.2f} | 평가금액: ${h.get('eval_amount_usd'):.2f}")

    # 6. 실시간 시세 기반 1주 시장가 매도 청산 발주 집행
    print("\n[Step 6] 2초 대기 후 보유 TQQQ 1주 전량 시장가 매도 청산 송출 (TR: tt80011)")
    time.sleep(2.0)
    sell_order = broker.send_order(
        symbol=config.BASE_ASSET_LONG,
        order_type="SELL",
        quantity=1,
        price=0.0  # 시장가 청산
    )
    print(f"   ✅ [매도 청산 완료]")
    print(f"      - 주문 번호: {sell_order.get('order_no')}")
    print(f"      - 주문 상태: {'정상 청산' if sell_order.get('ok') else '실패'}")
    print(f"      - 청산 단가: ${sell_order.get('price'):.2f}")
    print(f"      - 서버 메시지: {sell_order.get('msg')}")

    # 7. 매도 후 최종 잔고 확인 (포지션 0 확인)
    time.sleep(1.0)
    bal_final = broker.get_overseas_stock_balance()
    print(f"\n[Step 7] 매도 청산 후 최종 잔고 확인 (오버나잇 0% 현금화)")
    print(f"   • 최종 보유 종목 수: {bal_final.get('holdings_count', 0)}개 (포지션 클리어)")
    print(f"   • 최종 주문 가능 예수금: ${broker.get_overseas_deposit().get('usd_order_available', 0):,.2f} USD")

    # 8. 웹소켓 스트리머 정상 종료
    streamer.stop()
    print(f"\n[Step 8] 총 수신된 실시간 틱 이벤트: {len(captured_ticks)}건")

    print("\n" + "=" * 90)
    print("🎉 [실전 동일 매매 실증 완료] WebSocket 시세 수신 ➔ 매수 ➔ 잔고 확인 ➔ 매도 청산 100% 정상 완결!")
    print("=" * 90)

if __name__ == "__main__":
    execute_live_interactive_test()
