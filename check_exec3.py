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
        "qry_tp": "2" 
    }
    res = broker._send_tr_request(endpoint="/api/us/acnt", api_id="ust21050", body_dict=body, raise_on_error=False)
    print("qry_tp 2:", json.dumps(res, indent=2, ensure_ascii=False))

    body["qry_tp"] = "all"
    res = broker._send_tr_request(endpoint="/api/us/acnt", api_id="ust21050", body_dict=body, raise_on_error=False)
    print("qry_tp all:", json.dumps(res, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    run()
