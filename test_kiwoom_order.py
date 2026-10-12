import config
from core.kiwoom_broker import KiwoomBroker

def run_test_order():
    print("=== [TEST] Placing Kiwoom Virtual Order ===")
    
    # 1. Initialize broker explicitly in SIMULATION mode if possible, but we'll just instantiate
    # The current code might fall back to simulation if KIS_MODE=VIRTUAL is translated to kiwoom
    # Wait, let's just create it and test.
    broker = KiwoomBroker()
    
    # 2. Place a limit buy order for 1 share of TQQQ at $10.0 to ensure it doesn't accidentally fill right now
    try:
        res = broker.send_order(
            symbol="TQQQ",
            order_type="BUY",
            quantity=1,
            price=10.0,
            exchange="ND" # Or leave it as None to let it map automatically
        )
        print("✅ Order Request Successful:")
        print(res)
    except Exception as e:
        print("❌ Order Request Failed:")
        print(e)

if __name__ == "__main__":
    run_test_order()
