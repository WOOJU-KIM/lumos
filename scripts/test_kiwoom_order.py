import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.kiwoom_broker import KiwoomBroker
import config

def fire_test_order():
    broker = KiwoomBroker(is_simulation=True)
    print(f"[*] 키움증권(Kiwoom) API Broker Initialized. Mode: {broker.mode_str}")
    
    print("\n[1] SOXL 매수 주문 패킷(Payload) 및 체결 로직 테스트...")
    try:
        res_buy = broker.send_order(symbol="SOXL", order_type="BUY", quantity=1, price=1.0)
        print(f"BUY Result: {json.dumps(res_buy, ensure_ascii=False, indent=2)}")
    except Exception as e:
        print(f"BUY Failed: {e}")

    print("\n[2] SOXS 매수 주문 패킷(Payload) 및 체결 로직 테스트...")
    try:
        res_sell = broker.send_order(symbol="SOXS", order_type="BUY", quantity=1, price=1.0)
        print(f"SELL Result: {json.dumps(res_sell, ensure_ascii=False, indent=2)}")
    except Exception as e:
        print(f"SELL Failed: {e}")

if __name__ == "__main__":
    fire_test_order()
