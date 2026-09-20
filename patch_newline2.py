import os
from pathlib import Path

fpath = Path('scripts/live_trade_test_clean.py')
content = fpath.read_text(encoding='utf-8')

content = content.replace("f.write(test_log + '\\\\n')", "f.write(test_log + '\\n')")
content = content.replace("f.write(f\"\\\\n=====================================\\\\n\")", "f.write(f\"\\n=====================================\\n\")")
content = content.replace("f.write(f\"🚀 매매 테스트 시작: {feeder.target_date}\\\\n\")", "f.write(f\"🚀 매매 테스트 시작: {feeder.target_date}\\n\")")
content = content.replace("f.write(f\"=====================================\\\\n\")", "f.write(f\"=====================================\\n\")")

fpath.write_text(content, encoding='utf-8')
