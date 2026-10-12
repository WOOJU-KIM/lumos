import os, sys, time
from pathlib import Path
from unittest.mock import patch
from datetime import datetime as real_datetime

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["KIWOOM_IS_SIMULATION"] = "1"
import core.market_calendar
from scripts.run_mock_live import KiwoomLiveRunner

original_get_market_status = core.market_calendar.USMarketCalendar.get_market_status

def mock_get_market_status(*args, **kwargs):
    res = original_get_market_status(*args, **kwargs)
    res["is_open"] = True
    return res

core.market_calendar.USMarketCalendar.get_market_status = mock_get_market_status

def mock_verify_time_synchronization():
    return {
        "all_ok": True,
        "time_sync_ok": True,
        "switching_ok": True,
        "delta_hours": 13.0,
        "expected_diff": 13.0,
        "is_dst": True,
        "dst_text": "EDT",
        "now_kst_str": "MOCK KST",
        "now_ny_str": "MOCK NYT",
        "checklist_items": ["Mock item"]
    }

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
    print("Starting perfectly patched test...")
    with patch('core.market_calendar.datetime', MockDatetime), \
         patch('scripts.run_mock_live.datetime', MockDatetime), \
         patch('core.market_calendar.USMarketCalendar.get_market_status', side_effect=mock_get_market_status), \
         patch('core.market_calendar.USMarketCalendar.verify_time_synchronization', side_effect=mock_verify_time_synchronization), \
         patch('scripts.run_mock_live.time.time', mock_time_time), \
         patch('scripts.run_mock_live.time.sleep', mock_sleep):
         
         global_runner = KiwoomLiveRunner(is_simulation=True)
         
         def filtered_send_tg(message, event_time_str=None):
             if "15분 시그널" in message or "15m Signal" in message or "Briefing" in message or "진입 대기" in message or "정기 브리핑" in message:
                 return
             return original_send_tg(message, event_time_str=event_time_str)

         original_send_tg = global_runner.dispatcher.send_telegram_message
         global_runner.dispatcher.send_telegram_message = filtered_send_tg
         
         print("Start feeder thread manually so current_mock_time populates...")
         global_runner.ws_streamer.start()
         
         while global_runner.ws_streamer.current_mock_time is None:
             real_sleep(0.1)
             
         print(f"Feeder ready. First tick time: {global_runner.ws_streamer.current_mock_time}")
         
         try:
             global_runner.run_once_and_start_listener()
         except KeyboardInterrupt:
             pass
             
         while True:
             if getattr(global_runner.ws_streamer, "is_finished", False):
                 print("Mock feeder finished. Exiting test.")
                 break
             real_sleep(1)

if __name__ == '__main__':
    main()
