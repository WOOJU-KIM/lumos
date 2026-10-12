import os, sys, time, threading
from pathlib import Path
from unittest.mock import patch
from datetime import datetime as real_datetime

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["KIWOOM_IS_SIMULATION"] = "1"

from scripts.mock_live_runner import KiwoomLiveRunner
import core.market_calendar

original_get_market_status = core.market_calendar.USMarketCalendar.get_market_status

def mock_get_market_status(*args, **kwargs):
    res = original_get_market_status(*args, **kwargs)
    res["is_open"] = True
    return res

core.market_calendar.USMarketCalendar.get_market_status = mock_get_market_status

# Make MockDatetime so live_runner uses mock time
class MockDatetime(real_datetime):
    @classmethod
    def now(cls, tz=None):
        global global_runner
        if 'global_runner' in globals() and hasattr(global_runner, 'ws_streamer') and getattr(global_runner.ws_streamer, 'current_mock_time', None):
            mock_time = global_runner.ws_streamer.current_mock_time
            if hasattr(mock_time, 'to_pydatetime'):
                mock_time = mock_time.to_pydatetime()
            if tz is not None:
                from zoneinfo import ZoneInfo
                if mock_time.tzinfo is None:
                    mock_time = mock_time.replace(tzinfo=ZoneInfo("Asia/Seoul"))
                return mock_time.astimezone(tz)
            return mock_time
        return real_datetime.now(tz)

real_time_time = time.time
def mock_time_time():
    global global_runner
    if 'global_runner' in globals() and hasattr(global_runner, 'ws_streamer') and getattr(global_runner.ws_streamer, 'current_mock_time', None):
        return global_runner.ws_streamer.current_mock_time.timestamp()
    return real_time_time()

real_sleep = time.sleep
def mock_sleep(secs):
    if secs > 0:
        real_sleep(0.001)

def main():
    global global_runner
    print("Starting unblocked mock live test...")
    with patch('core.market_calendar.datetime', MockDatetime), \
         patch('scripts.mock_live_runner.datetime', MockDatetime), \
         patch('core.live_runner.datetime', MockDatetime), \
         patch('core.live_runner.time.time', mock_time_time), \
         patch('core.live_runner.time.sleep', mock_sleep):
         
         global_runner = KiwoomLiveRunner(is_simulation=True)
         global_runner.controller.listen_loop = lambda *args, **kwargs: None
         
         while global_runner.ws_streamer.current_mock_time is None:
             real_sleep(0.1)
             
         print("Feeder ready. Start runner.")
         try:
             global_runner.run_once_and_start_listener()
         except KeyboardInterrupt:
             pass
             
         while True:
             if getattr(global_runner.ws_streamer, "is_finished", False):
                 print("Feeder finished!")
                 break
             print(f"Current mock time: {global_runner.ws_streamer.current_mock_time}")
             real_sleep(1)

if __name__ == '__main__':
    main()
