import os
import json
from dotenv import load_dotenv

load_dotenv()
from core.kis_client import KISClient

def run():
    client = KISClient(mode="VIRTUAL") # or REAL depending on what they used
    res = client.inquire_ccnl(start_date="20260921", end_date="20260921")
    print("VIRTUAL:", json.dumps(res, indent=2, ensure_ascii=False))

    client_real = KISClient(mode="REAL")
    res_real = client_real.inquire_ccnl(start_date="20260921", end_date="20260921")
    print("REAL:", json.dumps(res_real, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    run()
