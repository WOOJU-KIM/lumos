import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.mock_kiwoom_feeder import MockKiwoomFeeder
from core.moe_orchestrator import MoEMetaOrchestrator
import config

def run_test_for_date(target_date):
    feeder = MockKiwoomFeeder(target_date=target_date)
    # Remove sleep from feeder to run fast
    import time
    feeder.start()
    
    last_eval_time = None
    scores = []
    
    while True:
        if getattr(feeder, "is_finished", False):
            break
            
        current_time = feeder.current_mock_time
        if current_time is None:
            time.sleep(0.001)
            continue
            
        if current_time.minute % 15 == 0 and last_eval_time != current_time:
            last_eval_time = current_time
            df_15m = feeder.data_lake.load_candles("TQQQ", "15m")
            if df_15m.empty: continue
            
            live_prices = {sym: feeder.get_latest_price(sym, 0.0) for sym in config.ALL_SYMBOLS}
            
            moe = MoEMetaOrchestrator()
            moe_res = moe.evaluate_dual_filter_signal(df_candle_15m=df_15m, live_prices=live_prices)
            
            conf = float(moe_res.get('gating_confidence', 0.0)) * 100.0
            direction = moe_res.get('direction', 'NONE')
            
            scores.append(f"[{current_time.strftime('%Y-%m-%d %H:%M')}] Dir: {direction:<10} Conf: {conf:>5.2f}%")
                
    return scores

if __name__ == "__main__":
    dates = ["2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18"]
    print("=== REFACTOR #1 AFTER SCORES ===")
    for d in dates:
        print(f"Testing {d}...")
        scores = run_test_for_date(d)
        for s in scores:
            print("   " + s)
