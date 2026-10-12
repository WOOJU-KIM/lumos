import sys
import json
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.kiwoom_broker import KiwoomBroker

kb = KiwoomBroker(is_simulation=False)
token = kb.get_access_token()

endpoints = [
    "/api/us/trade",
    "/api/us/stock/order",
    "/api/trade/us/order",
    "/api/trade/order",
    "/api/order",
    "/api/us/acnt",
    "/api/us/quote",
    "/api/us/order",
    "/api/overseas/order",
    "/api/overseas-stock/order",
    "/api/v1/us/order",
    "/api/v1/trade/order",
]

api_ids = [
    "tt80010", "tt80011", "tt31001", "tt31002", "ust80010", "ust80011", 
    "ust21010", "ust21020", "ust21030", "ust21040", "ust21050", "ust21070",
    "ust21110", "ust10000", "kt00001", "kt00002"
]

print("=" * 80)
print("🔍 [키움 엔드포인트 & TR ID 전수 지원 테이블 스캔]")
print("=" * 80)

# Check what API IDs are supported on /api/us/* endpoints
for ep in ["/api/us/order", "/api/us/trade", "/api/us/acnt", "/api/us/quote"]:
    for aid in api_ids:
        url = f"{kb.base_url}{ep}"
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "authorization": f"Bearer {token}",
            "api-id": aid,
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
            with urllib.request.urlopen(req, timeout=3) as response:
                res = json.loads(response.read().decode("utf-8"))
                ret_code = res.get("return_code")
                msg = res.get("return_msg")
                if "해당 URI에서는 지원하는 API ID가 아닙니다" not in msg:
                    print(f"🎉 [발견!] {ep} | {aid} -> return_code={ret_code}, msg={msg}")
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8")
            if "지원하는 API ID가 아닙니다" not in err:
                print(f"💡 [응답!] {ep} | {aid} -> HTTP {e.code}: {err[:80]}")
        except Exception:
            pass

print("=" * 80)
