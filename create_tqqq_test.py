import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config

with open('scripts/run_specific_week.py', 'r', encoding='utf-8') as f:
    c = f.read()

# I want to make sure it runs TQQQ/SQQQ
# But wait, config is currently SOXL. I should dynamically override config inside the script just for testing.
override = """
    config.BASE_ASSET_LONG = 'TQQQ'
    config.BASE_ASSET_SHORT = 'SQQQ'
    long_15m = lake.load_candles(config.BASE_ASSET_LONG, "15m")
"""
c = c.replace('long_15m = lake.load_candles(config.BASE_ASSET_LONG, "15m")', override)

with open('scripts/run_specific_week_tqqq.py', 'w', encoding='utf-8') as f:
    f.write(c)

print("Created run_specific_week_tqqq.py")
