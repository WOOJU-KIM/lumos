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
        "qry_tp": "1" 
    }
    res = broker._send_tr_request(endpoint="/api/us/acnt", api_id="ust21050", body_dict=body, raise_on_error=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    run()
