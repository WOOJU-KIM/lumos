import config
from core.kis_client import KISClient

def run_test_order():
    print("=== [TEST] Placing Virtual Order ===")
    
    # 1. Initialize client explicitly in VIRTUAL mode for safety
    client = KISClient(mode="VIRTUAL")
    print(f"Current KIS Mode: {client.mode}")
    
    # 2. Place a limit buy order for 1 share of TQQQ at a very low price to ensure it doesn't accidentally fill right now
    # TQQQ is currently around $60~80+, so $10 limit is safe.
    # ord_dvsn="00" means 지정가 (Limit Order)
    try:
        res = client.order_overseas_stock(
            ticker="TQQQ",
            order_type="buy",
            qty=1,
            price=10.0,
            ord_dvsn="00" 
        )
        print("✅ Order Request Successful:")
        print(res)
    except Exception as e:
        print("❌ Order Request Failed:")
        print(e)

if __name__ == "__main__":
    run_test_order()
