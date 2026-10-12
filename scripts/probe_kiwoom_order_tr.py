import sys
import os
import config
import json
import urllib.request
from pathlib import Path

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.kiwoom_broker import KiwoomBroker

kb = KiwoomBroker(is_simulation=False)
token = kb.get_access_token()

candidate_apis = [
    ("/api/us/order", "tt31001"),
    ("/api/us/order", "tt31002"),
    ("/api/us/order", "ust80010"),
    ("/api/us/order", "ust80011"),
    ("/api/us/order", "ust31001"),
    ("/api/us/order", "ust31002"),
    ("/api/us/order", "tt80010"),
    ("/api/dpt/overseas-stock/v1/trading/order", "JTT80010"),
    ("/api/dpt/overseas-stock/v1/trading/order", "TT80010"),
    ("/api/dpt/overseas-stock/v1/trading/order", "VTTC0802U"),
]

print("=" * 80)
print("🔍 [키움 해외주식 주문 TR ID 및 엔드포인트 전수 매핑 진단]")
print("=" * 80)

for endpoint, api_id in candidate_apis:
    url = f"{kb.base_url}{endpoint}"
    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "authorization": f"Bearer {token}",
        "api-id": api_id,
        "cont-yn": "N",
        "next-key": ""
    }
    body = {
        "cano": kb.account_no,
        "acnt_prdt_cd": kb.account_type,
        "symb": config.BASE_ASSET_LONG,
        "excd": "NAS",
        "ord_qty": "1",
        "ord_unpr": "0",
        "ord_dv": "01"
    }
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            res = json.loads(response.read().decode("utf-8"))
            print(f"✅ [{endpoint} | {api_id}] -> HTTP 200, return_code={res.get('return_code')}, msg={res.get('return_msg')}")
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8")
        print(f"❌ [{endpoint} | {api_id}] -> HTTP {e.code}: {err[:90]}")
    except Exception as ex:
        print(f"⚠️ [{endpoint} | {api_id}] -> {ex}")

print("=" * 80)
