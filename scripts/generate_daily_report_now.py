import os
import sys

# Add lumos path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.kiwoom_broker import KiwoomBroker
from core.eod_reporter import DailyReporter
from core.system_logger import system_logger

if __name__ == "__main__":
    print("Initializing KiwoomBroker (Simulation Mode for safety)...")
    # For a real run without actual trades, we can just get balance. 
    # Simulation mode should be safe to fetch balance or we can use real mode if needed.
    # We will try simulation first.
    broker = KiwoomBroker(is_simulation=True)
    
    print("Testing connection and fetching broker report...")
    try:
        report = broker.get_official_broker_report()
        print("Broker connection successful. Current Balance (KRW):", report.get("total_eval_krw"))
    except Exception as e:
        print("Failed to get real broker report (maybe not logged in?):", e)
        print("Falling back to dummy data for demonstration.")
        class FallbackBroker:
            def get_official_broker_report(self):
                return {
                    "total_eval_krw": 54000000,
                    "realized_rate_pct": 1.25
                }
        broker = FallbackBroker()
        
    print("Generating Daily Report...")
    reporter = DailyReporter(broker)
    res = reporter.generate_report()
    
    print("\n[Report Generated Successfully]")
    import json
    print(json.dumps(res, indent=2, ensure_ascii=False))
