import os
import json
from dotenv import load_dotenv

load_dotenv()
from core.kiwoom_broker import KiwoomBroker

def run():
    broker = KiwoomBroker(is_simulation=False) # 0 for real if requested, but let's see what happens with default
    print("Broker Mode:", broker.mode_str)
    res = broker.get_daily_execution_history()
    print(json.dumps(res, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    run()
