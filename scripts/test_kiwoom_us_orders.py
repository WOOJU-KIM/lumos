import urllib.request
import urllib.error
import json
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.kiwoom_broker import KiwoomBroker

broker = KiwoomBroker(is_simulation=True)
token = broker.get_access_token()

url = f"{broker.base_url}/api/us/ordr"
headers = {
    "Content-Type": "application/json;charset=UTF-8",
    "authorization": f"Bearer {token}",
    "api-id": "ust20000",
    "cont-yn": "N",
    "next-key": ""
}

test_cases = [
    ("ND", "NVDA", "216.85"),
    ("ND", "QQQ", "710.00"),
    ("NA", "TQQQ", "122.00"),
    ("NY", "TQQQ", "122.00"),
    ("ND", "TQQQ", "122.00"),
    ("AMS", "TQQQ", "122.00"),
    ("ND", "SQQQ", "45.50")
]

print("=" * 80)
print(f"🔍 [Kiwoom Mock Order TR: ust20000 Direct Test]")
print("=" * 80)

for stex, stk, px in test_cases:
    time.sleep(1.5)
    body = {
        "cano": broker.account_no,
        "acnt_prdt_cd": "01",
        "stex_tp": stex,
        "stk_cd": stk,
        "ord_qty": "1",
        "ord_uv": px,
        "trde_tp": "00"  # 00: 지정가
    }
    
    print(f"\n>>> [REQUEST: {stex} | {stk} | ${px}]")
    print(f"URL: {url}")
    print(f"Headers: {headers}")
    print(f"Body: {json.dumps(body)}")

    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"<<< [RESPONSE {resp.status}] Code: {data.get('return_code')} | Msg: {data.get('return_msg')} | Full: {data}")
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8")
        print(f"<<< [HTTP ERROR {e.code}] {err_msg}")
    except Exception as e:
        print(f"<<< [EXCEPTION] {e}")
