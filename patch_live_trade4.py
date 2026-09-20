import os
from pathlib import Path

fpath = Path('scripts/live_trade_test.py')
content = fpath.read_text(encoding='utf-8')

# The first instance is inside USMarketCalendar.get_market_status
# Let's fix that specific method
idx = content.find('def get_market_status')
if idx != -1:
    end_idx = content.find('def verify_time_synchronization', idx)
    func_str = content[idx:end_idx]
    new_func_str = func_str.replace('now_dt = getattr(self.ws_streamer, "current_mock_time", datetime.now()) or datetime.now()', 'now_dt = now_dt or datetime.now()')
    content = content[:idx] + new_func_str + content[end_idx:]

fpath.write_text(content, encoding='utf-8')
