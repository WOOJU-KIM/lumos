import os
from pathlib import Path

fpath = Path('scripts/live_trade_test.py')
content = fpath.read_text(encoding='utf-8')

idx = content.find('if __name__ == "__main__":')
if idx != -1:
    content = content[:idx] + """if __name__ == "__main__":
    is_sim = True
    print("🚀 시작: 매매 테스트 (Mock Live Runner)")
    runner = KiwoomLiveRunner(is_simulation=is_sim)
    runner.run_once_and_start_listener()
"""

fpath.write_text(content, encoding='utf-8')
