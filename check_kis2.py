import os
import json
from dotenv import load_dotenv

load_dotenv()
from core.kis_client import KisClient

def run():
    try:
        client = KisClient(mode="VIRTUAL") 
        res = client.inquire_ccnl(start_date="20260921", end_date="20260921")
        print("VIRTUAL:", json.dumps(res, indent=2, ensure_ascii=False))
    except Exception as e: print(e)

    try:
        client_real = KisClient(mode="REAL")
        res_real = client_real.inquire_ccnl(start_date="20260921", end_date="20260921")
        print("REAL:", json.dumps(res_real, indent=2, ensure_ascii=False))
    except Exception as e: print(e)

if __name__ == '__main__':
    run()
