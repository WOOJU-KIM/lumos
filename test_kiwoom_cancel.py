import config
from core.kiwoom_broker import KiwoomBroker

def run_test_cancel():
    print("=== [TEST] Canceling all open Kiwoom Virtual Orders ===")
    broker = KiwoomBroker()
    
    try:
        res = broker.cancel_all_open_orders(symbol="TQQQ")
        print("✅ Cancel Request Output:")
        print(res)
    except Exception as e:
        print("❌ Cancel Request Failed:")
        print(e)

if __name__ == "__main__":
    run_test_cancel()
