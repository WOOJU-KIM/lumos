import sys, pandas as pd, sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from core.data_lake import MarketDataLake
lake = MarketDataLake()
try:
    lake.load_candles("NVDA", "15m", end_dt={"this": "is_a_dict"})
except Exception as e:
    print(f"Exception: {e}")
