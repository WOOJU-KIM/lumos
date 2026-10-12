import sys
from pathlib import Path

bt_path = Path("c:/Users/chabo/OneDrive/바탕 화면/lumos/core/backtest_engine.py")
content = bt_path.read_text(encoding="utf-8")

# I need to modify the run_backtest and _execute_triple_screen_simulation methods.
