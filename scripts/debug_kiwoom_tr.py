import sys
import json
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.kiwoom_broker import KiwoomBroker

broker = KiwoomBroker()
token = broker.get_access_token()

print("=" * 80)
print(f"🔍 [키움증권 REST TR 및 엔드포인트 전수 진단 - Base: {broker.base_url}]")
print("=" * 80)

test_cases = [
    ("/api/us/quote", "ust10000", {"symb": config.BASE_ASSET_LONG, "excd": "NAS"}),
    ("/api/us/acnt", "ust21110", {"cano": broker.account_no, "acnt_prdt_cd": "01"}),
    ("/api/us/acnt", "ust21070", {"cano": broker.account_no, "acnt_prdt_cd": "01", "qry_tp": "1"}),
    ("/api/us/acnt", "ust21050", {"cano": broker.account_no, "acnt_prdt_cd": "01", "qry_tp": "0"}),
    ("/api/us/order", "tt80010", {"cano": broker.account_no, "acnt_prdt_cd": "01", "symb": config.BASE_ASSET_LONG, "excd": "NAS", "ord_qty": "1", "ord_unpr": "0", "ord_dv": "01"}),
    ("/api/us/order", "ust80010", {"cano": broker.account_no, "acnt_prdt_cd": "01", "symb": config.BASE_ASSET_LONG, "excd": "NAS", "ord_qty": "1", "ord_unpr": "0", "ord_dv": "01"}),
]

for ep, aid, body in test_cases:
    url = f"{broker.base_url}{ep}"
    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "authorization": f"Bearer {token}",
        "api-id": aid,
        "cont-yn": "N",
        "next-key": ""
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            rc = data.get("return_code")
            rm = data.get("return_msg")
            print(f"[{ep} | {aid}] -> Status {resp.status} | ReturnCode: {rc} | Msg: {rm}")
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8")
        print(f"[{ep} | {aid}] -> HTTP Error {e.code}: {err}")
    except Exception as e:
        print(f"[{ep} | {aid}] -> Exception: {e}")
