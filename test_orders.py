import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.kiwoom_broker import KiwoomBroker
import config
import json

broker = KiwoomBroker()
print("============== [Kiwoom Order Test] ==============")
print(f"Broker Mode: {broker.mode_str}")
print(f"Target Symbol: SOXL (Mapped to NY Exchange internally)")

# 1. Test BUY
print("\n[1] Sending BUY order for 1 SOXL at $10.00...")
try:
    buy_res = broker.send_order(symbol="SOXL", order_type="BUY", quantity=1, price=10.00)
    print(json.dumps(buy_res, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"BUY Exception caught correctly: {e}")

# 2. Test SELL
print("\n[2] Sending SELL order for 1 SOXS at $10.00...")
try:
    sell_res = broker.send_order(symbol="SOXS", order_type="SELL", quantity=1, price=10.00)
    print(json.dumps(sell_res, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"SELL Exception caught correctly: {e}")
