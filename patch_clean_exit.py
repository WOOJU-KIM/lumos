import os
from pathlib import Path

fpath = Path('scripts/live_trade_test_clean.py')
content = fpath.read_text(encoding='utf-8')

content = content.replace(
    'if current_time is None:',
    'if getattr(feeder, "is_finished", False):\n            print("✅ 시뮬레이션 종료")\n            break\n\n        if current_time is None:'
)

fpath.write_text(content, encoding='utf-8')
