import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.kis_client import KisClient

def test_order():
    client = KisClient(mode="VIRTUAL")
    print(f"KIS Client Mode: {client.mode}")
    
    # Try NASD for SOXL
    print("\n--- 1. Testing SOXL with NASD ---")
    client.get_exchange_code = lambda x: "NASD"
    res1 = client.order_overseas_stock(ticker="SOXL", order_type="BUY", price=1.0, qty=1)
    print(f"Result (NASD): {res1}")

    time.sleep(1)

    # Try NYSE for SOXL
    print("\n--- 2. Testing SOXL with NYSE ---")
    client.get_exchange_code = lambda x: "NYSE"
    res2 = client.order_overseas_stock(ticker="SOXL", order_type="BUY", price=1.0, qty=1)
    print(f"Result (NYSE): {res2}")

    time.sleep(1)

    # Try AMEX for SOXL
    print("\n--- 3. Testing SOXL with AMEX ---")
    client.get_exchange_code = lambda x: "AMEX"
    res3 = client.order_overseas_stock(ticker="SOXL", order_type="BUY", price=1.0, qty=1)
    print(f"Result (AMEX): {res3}")

if __name__ == "__main__":
    test_order()
