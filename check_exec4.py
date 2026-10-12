import os
import json
from dotenv import load_dotenv

load_dotenv()
from core.kiwoom_broker import KiwoomBroker

def run():
    broker = KiwoomBroker(is_simulation=False) 
    body = {
        "cano": broker.account_no,
        "acnt_prdt_cd": broker.account_type,
        "qry_tp": "0" 
    }
    
    for tr in ["ust21040", "ust21060", "ust21030", "ust21080"]:
        print(f"\\n--- Testing {tr} ---")
        res = broker._send_tr_request(endpoint="/api/us/acnt", api_id=tr, body_dict=body, raise_on_error=False)
        print(json.dumps(res, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    run()
