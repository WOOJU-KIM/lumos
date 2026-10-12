import sys
import json
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.kiwoom_broker import KiwoomBroker

kb = KiwoomBroker(is_simulation=False)
token = kb.get_access_token()

order_api_candidates = [
    # US Order TRs
    "tt80010", "tt80011", "tt80012",
    "tt10001", "tt10002", "tt10003",
    "tt30001", "tt30002", "tt30003",
    "tt31001", "tt31002", "tt31003",
    "ust80010", "ust80011", "ust80012",
    "ust10001", "ust10002", "ust10003",
    "ust30001", "ust30002", "ust30003",
    "ust31001", "ust31002", "ust31003",
    "ust21010", "ust21020", "ust21030",
    "ot80010", "ot80011", "ot80012",
    "ot31001", "ot31002", "ot31003",
    "vt80010", "vt80011", "vt80012",
    "kt00001", "kt00002", "kt00003",
    "opw00001", "opw00002", "opw00004", "opw00005"
]

print("=" * 80)
print("🔍 [키움 /api/us/order 및 /api/order TR ID 전수 프로빙]")
print("=" * 80)

for ep in ["/api/us/order", "/api/order", "/api/us/trade", "/api/trade"]:
    for aid in order_api_candidates:
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
                msg = res.get("return_msg", "")
                if "지원하는 API ID가 아닙니다" not in msg:
                    print(f"🎉 [발견 성공!] {ep} | {aid} -> return_code={ret_code}, msg={msg}")
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8")
            if "지원하는 API ID가 아닙니다" not in err and "404" not in str(e.code):
                print(f"💡 [응답!] {ep} | {aid} -> HTTP {e.code}: {err[:80]}")
        except Exception:
            pass

print("=" * 80)
