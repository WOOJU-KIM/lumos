import os
import sys
import time
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.kiwoom_broker import KiwoomBroker
from core.system_logger import system_logger

def run_test():
    print("🚀 단일 TQQQ 매수/매도 테스트 시작...")
    broker = KiwoomBroker(is_simulation=True)
    
    # 연결 테스트 및 예수금 조회
    conn_res = broker.test_connection()
    if not conn_res.get("ok"):
        print(f"❌ 연결 실패: {conn_res.get('msg')}")
        return
        
    print(f"✅ 접속 완료. 가용 예수금: ${conn_res.get('usd_order_available', 0.0)}")
    
    symbol = "TQQQ"
    
    # 현재가 조회
    quote = broker.get_stock_quote(symbol)
    cur_px = float(quote.get('last_price', 0.0))
    print(f"📈 {symbol} 현재가: ${cur_px}")
    
    if cur_px <= 0:
        print("❌ 현재가 조회 실패. 거래를 중단합니다.")
        return
        
    # 1주 매수 시도
    print("\n🛒 1주 매수 주문 발송 중...")
    try:
        buy_res = broker.send_order(symbol=symbol, order_type="BUY", quantity=1, price=0.0)
        print(f"✅ 매수 주문 성공: {json.dumps(buy_res, indent=2, ensure_ascii=False)}")
        buy_order_no = buy_res.get("order_no")
    except Exception as e:
        print(f"❌ 매수 주문 에러: {e}")
        return
        
    # 미체결 및 잔고 대기
    print("\n⏳ 5초 대기 (체결 확인)...")
    time.sleep(5)
    
    open_ords = broker.get_open_orders()
    print(f"📋 미체결 내역: {json.dumps(open_ords.get('orders', []), ensure_ascii=False)}")
    
    stk_bal = broker.get_overseas_stock_balance(force_refresh=True)
    holdings = stk_bal.get('holdings', [])
    tqqq_held = next((h for h in holdings if h.get("symbol", "").upper() == symbol), None)
    
    if tqqq_held:
        qty = int(float(str(tqqq_held.get("quantity", 0)).replace(",", "")))
        print(f"✅ 잔고 확인됨: {symbol} {qty}주 보유 중")
    else:
        print("⚠️ 아직 잔고에 반영되지 않았거나 미체결 상태일 수 있습니다. (그래도 매도 시도 진행)")
        
    # 1주 매도 시도
    print("\n💸 1주 매도 주문 발송 중...")
    try:
        sell_res = broker.send_order(symbol=symbol, order_type="SELL", quantity=1, price=0.0)
        print(f"✅ 매도 주문 성공: {json.dumps(sell_res, indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"❌ 매도 주문 에러: {e}")
        return
        
    print("\n⏳ 5초 대기 (체결 확인)...")
    time.sleep(5)
    stk_bal = broker.get_overseas_stock_balance(force_refresh=True)
    holdings = stk_bal.get('holdings', [])
    tqqq_held = next((h for h in holdings if h.get("symbol", "").upper() == symbol), None)
    if not tqqq_held or int(float(str(tqqq_held.get("quantity", 0)).replace(",", ""))) == 0:
         print(f"✅ 매도 완료 및 잔고 소진 확인.")
    else:
         print(f"⚠️ 매도 후 잔고가 남아 있습니다: {tqqq_held}")

    print("\n🎉 단일 틱 매매 테스트 종료.")

if __name__ == "__main__":
    run_test()
