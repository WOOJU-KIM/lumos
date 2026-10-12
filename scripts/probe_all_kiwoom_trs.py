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

endpoints = [
    "/api/us/order",
    "/api/us/acnt",
    "/api/us/quote",
    "/api/us/trd",
    "/api/us/trade",
    "/api/dpt/overseas-stock/order",
    "/api/dpt/overseas-stock/trading",
    "/api/overseas-stock/order"
]

tr_ids = [
    "ust21070", "ust21110", "ust21050",
    "tt80010", "tt80011", "ust80010", "ust80011",
    "ust10000", "ust10001", "ust01000",
    "kt10000", "kt10001", "kt00001", "kt00018",
    "OS_ORD_BUY", "OS_ORD_SELL", "ORD_BUY", "ORD_SELL",
    "vt80010", "vt80011", "vt10000", "ut80010", "ut80011"
]

print("=" * 80)
print(f"🔍 [키움 OpenAPI TR & URI 전수 브루트포스 매핑 탐색]")
print("=" * 80)

matches = []

for ep in endpoints:
    for tr in tr_ids:
        url = f"{broker.base_url}{ep}"
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "authorization": f"Bearer {token}",
            "api-id": tr,
            "cont-yn": "N",
            "next-key": ""
        }
        body = {
            "cano": broker.account_no,
            "acnt_prdt_cd": "01",
            "symb": config.BASE_ASSET_LONG,
            "excd": "NAS",
            "ord_qty": "1",
            "ord_unpr": "0",
            "ord_dv": "01",
            "qry_tp": "1"
        }
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                rc = data.get("return_code")
                rm = data.get("return_msg")
                if "1504" not in rm:
                    print(f"🎯 [MATCH!] Endpoint: {ep} | TR: {tr} ➔ RC: {rc} | Msg: {rm}")
                    matches.append((ep, tr, rc, rm))
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8")
            if "1504" not in err and "404" not in err:
                print(f"⚡ [HTTP {e.code}] Endpoint: {ep} | TR: {tr} ➔ {err}")
        except Exception:
            pass

print("=" * 80)
print(f"총 발견된 유효 엔드포인트-TR 조합: {len(matches)}개")
for m in matches:
    print(m)
print("=" * 80)
