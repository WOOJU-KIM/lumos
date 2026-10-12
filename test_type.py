import sys, pandas as pd, sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from core.mock_kiwoom_feeder import MockKiwoomFeeder
feeder = MockKiwoomFeeder(target_date='2026-09-21')
feeder.start()
import time
last_eval = None
while True:
    if getattr(feeder, 'is_finished', False): break
    mt = feeder.current_mock_time
    if mt is None: 
        time.sleep(0.01)
        continue
    if mt.minute % 15 == 0 and last_eval != mt:
        last_eval = mt
        df_15m = feeder.data_lake.load_candles('TQQQ', '15m')
        if not df_15m.empty:
            v = df_15m.iloc[-1]['datetime']
            print(f"type: {type(v)}, value: {v}")
            break
    time.sleep(0.001)
