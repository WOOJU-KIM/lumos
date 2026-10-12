import os
import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.kiwoom_broker import KiwoomBroker

def test_all_symbols():
    broker = KiwoomBroker(is_simulation=True)
    print(f"[*] 키움증권(Kiwoom) API Broker Initialized. Mode: {broker.mode_str}")
    
    test_symbols = ["SOXL", "SOXS", "QQQ", "TQQQ", "SQQQ", "NVDA", "VIXY", "IEF", "SOXX", "SPY"]
    
    print("\n[🚀 전 종목 거래소 라우팅 테스트 시작]")
    for sym in test_symbols:
        print(f"\n--- Testing Symbol: {sym} ---")
        try:
            # Send mock order
            res = broker.send_order(symbol=sym, order_type="BUY", quantity=1, price=1.0)
            
            # Since KIS mock server rejects it because it's closed, we just want to see the exchange code sent.
            # We can parse the exact exchange code from the result or just rely on broker logs.
            print(f"Result for {sym}: {json.dumps(res, ensure_ascii=False)}")
        except Exception as e:
            # We expect an exception because raise_on_error=True and market is closed
            print(f"Order for {sym} caught expected rejection: {e}")
            
        # sleep slightly to avoid aggressive rate limit on mock server
        time.sleep(0.5)

if __name__ == "__main__":
    test_all_symbols()
