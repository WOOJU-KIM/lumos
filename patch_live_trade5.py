import os
from pathlib import Path

fpath = Path('scripts/live_trade_test.py')
content = fpath.read_text(encoding='utf-8')

content = content.replace(
    'mkt_status = USMarketCalendar.get_market_status()',
    'mkt_status = USMarketCalendar.get_market_status(getattr(self.ws_streamer, "current_mock_time", None))'
)
content = content.replace(
    'mkt = USMarketCalendar.get_market_status()',
    'mkt = USMarketCalendar.get_market_status(getattr(self.ws_streamer, "current_mock_time", None))'
)
# Also change now_dt in AI Evaluation & Briefing
content = content.replace(
    'now_dt = datetime.now()',
    'now_dt = getattr(self.ws_streamer, "current_mock_time", datetime.now()) or datetime.now()'
)
content = content.replace(
    'now_t = time.time()',
    'now_t = now_dt.timestamp()'
)

fpath.write_text(content, encoding='utf-8')
