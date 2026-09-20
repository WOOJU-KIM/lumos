import os
from pathlib import Path

fpath = Path('scripts/live_trade_test_clean.py')
content = fpath.read_text(encoding='utf-8')

content = content.replace("f.write(test_log + '\\\\n')", "f.write(test_log + '\\n')")

fpath.write_text(content, encoding='utf-8')

# 덤으로 기존 로그 파일 깔끔하게 지우기
log_path = Path('data/trade_test_logs.txt')
if log_path.exists():
    log_path.unlink()
