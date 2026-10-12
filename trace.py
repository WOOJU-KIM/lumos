import sys, pandas as pd, sqlite3, traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from core.mock_kiwoom_feeder import MockKiwoomFeeder
from core.moe_orchestrator import MoEMetaOrchestrator
import config
import time

feeder = MockKiwoomFeeder(target_date='2026-09-21')
orchestrator = MoEMetaOrchestrator()
feeder.start()
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
            prices = {sym: feeder.get_latest_price(sym, 0.0) for sym in config.ALL_SYMBOLS}
            import sys
            import core.moe_orchestrator
            old_print = print
            def my_print(*args, **kwargs):
                if "CROSS_ASSET EXCEPTION" in args:
                    traceback.print_stack()
                old_print(*args, **kwargs)
            # monkey patch print in moe_orchestrator
            core.moe_orchestrator.print = my_print
            try:
                res = orchestrator.evaluate_dual_filter_signal(df_15m, prices)
            except Exception as e:
                traceback.print_exc()
            break
    time.sleep(0.001)
