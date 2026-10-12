import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.kiwoom_broker import KiwoomBroker

def main():
    print("🚀 실전 매수/매도 시그널 및 거래소(NY/ND) 구분 테스트 시작")
    broker = KiwoomBroker(is_simulation=True)
    
    print("\n--- 1. [NY 거래소] 종목 테스트 (SOXL) ---")
    ny_symbol = "SOXL"
    quote_ny = broker.get_stock_quote(ny_symbol)
    cur_px_ny = float(quote_ny.get("last_price", 30.0))
    print(f"[주가 수신] {ny_symbol} 현재가: ${cur_px_ny}")
    
    # 0.01달러 매수 주문 (체결 안 되게 매우 낮게)
    order_px_ny = 0.01 
    print(f"[매수 발주] {ny_symbol} 1주 @ ${order_px_ny} (NY 거래소 확인)")
    try:
        res_ny = broker.send_order(symbol=ny_symbol, order_type="BUY", quantity=1, price=order_px_ny)
        print(f"-> 결과: {res_ny}")
    except Exception as e:
        print(f"-> 예외 발생 (정상, 장시작전 등): {e}")
    
    time.sleep(1)
    
    print("\n--- 2. [ND 거래소] 종목 테스트 (QQQ) ---")
    nd_symbol = "QQQ"
    quote_nd = broker.get_stock_quote(nd_symbol)
    cur_px_nd = float(quote_nd.get("last_price", 400.0))
    print(f"[주가 수신] {nd_symbol} 현재가: ${cur_px_nd}")
    
    # 0.01달러 매수 주문 (체결 안 되게 매우 낮게)
    order_px_nd = 0.01 
    print(f"[매수 발주] {nd_symbol} 1주 @ ${order_px_nd} (ND 거래소 확인)")
    try:
        res_nd = broker.send_order(symbol=nd_symbol, order_type="BUY", quantity=1, price=order_px_nd)
        print(f"-> 결과: {res_nd}")
    except Exception as e:
        print(f"-> 예외 발생 (정상, 장시작전 등): {e}")
        
    print("\n--- 3. [NY 거래소] TQQQ/SQQQ 테스트 (TQQQ) ---")
    tqqq_symbol = "TQQQ"
    quote_tqqq = broker.get_stock_quote(tqqq_symbol)
    cur_px_tqqq = float(quote_tqqq.get("last_price", 40.0))
    print(f"[주가 수신] {tqqq_symbol} 현재가: ${cur_px_tqqq}")
    
    try:
        res_tqqq = broker.send_order(symbol=tqqq_symbol, order_type="BUY", quantity=1, price=order_px_nd)
        print(f"-> 결과: {res_tqqq}")
    except Exception as e:
        print(f"-> 예외 발생 (정상, 장시작전 등): {e}")

if __name__ == "__main__":
    main()
