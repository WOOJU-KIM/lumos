import os
import sys

# Add lumos path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.eod_reporter import DailyReporter

class MockBroker:
    def get_official_broker_report(self):
        return {
            "total_eval_krw": 54000000,
            "realized_rate_pct": 1.25
        }

if __name__ == "__main__":
    broker = MockBroker()
    reporter = DailyReporter(broker)
    res = reporter.generate_report()
    print("Report generated:", res)
