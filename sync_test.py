import os, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["KIWOOM_IS_SIMULATION"] = "1"
os.environ["LIVE_RUNNER_SILENT"] = "1"

from scripts.run_mock_live import KiwoomLiveRunner
import core.market_calendar
from core.kiwoom_broker import KiwoomBroker

original_get_market_status = core.market_calendar.USMarketCalendar.get_market_status
def mock_get_market_status(*args, **kwargs):
    res = original_get_market_status(*args, **kwargs)
    res["is_open"] = True
    return res
core.market_calendar.USMarketCalendar.get_market_status = mock_get_market_status

def mock_verify_time_synchronization():
    return {"all_ok": True, "time_sync_ok": True, "switching_ok": True, "delta_hours": 13.0, "expected_diff": 13.0, "is_dst": True, "dst_text": "EDT", "now_kst_str": "MOCK", "now_ny_str": "MOCK", "checklist_items": []}

runner = KiwoomLiveRunner(is_simulation=True)
from unittest.mock import patch
from datetime import datetime as real_datetime

class MockDatetime(real_datetime):
    @classmethod
    def now(cls, tz=None):
        mock_time = runner.ws_streamer.current_mock_time
        if mock_time:
            if hasattr(mock_time, 'to_pydatetime'): mock_time = mock_time.to_pydatetime()
            if tz:
                from zoneinfo import ZoneInfo
                if mock_time.tzinfo is None: mock_time = mock_time.replace(tzinfo=ZoneInfo("Asia/Seoul"))
                return mock_time.astimezone(tz)
            return mock_time
        return real_datetime.now(tz)

def mock_time_time():
    if runner.ws_streamer.current_mock_time:
        return runner.ws_streamer.current_mock_time.timestamp()
    return time.time()

# To perfectly sync, we will manually drive the feeder and runner in the SAME THREAD!
def main():
    print("🚀 시작: 9월 21일 동기화 매매 테스트")
    
    with patch('core.market_calendar.datetime', MockDatetime), \
         patch('core.live_runner.datetime', MockDatetime), \
         patch('scripts.run_mock_live.datetime', MockDatetime), \
         patch('core.market_calendar.USMarketCalendar.get_market_status', side_effect=mock_get_market_status), \
         patch('core.market_calendar.USMarketCalendar.verify_time_synchronization', side_effect=mock_verify_time_synchronization), \
         patch('core.live_runner.time.time', mock_time_time), \
         patch('core.live_runner.time.sleep', lambda x: None):
         
         # Telegram Mock
         def filtered_send_tg(msg, event_time_str=None):
             if "브리핑" in msg or "15분 시그널" in msg or "Briefing" in msg: return
             print(f"\n🔔 [텔레그램]\n{msg}")
         runner.dispatcher.send_telegram_message = filtered_send_tg
         
         runner.ws_streamer.start() # Starts thread, but we will patch sleep
         
         # Wait for feeder to load
         while runner.ws_streamer.current_mock_time is None:
             time.sleep(0.1)
             
         print("Feeder started.")
         # Force first run setup
         runner._last_market_session = None
         
         # Now we just poll
         while not getattr(runner.ws_streamer, "is_finished", False):
             try:
                 runner._market_execution_loop_once() # We need to create this!
             except AttributeError:
                 pass
             time.sleep(0.001)

if __name__ == '__main__':
    main()
