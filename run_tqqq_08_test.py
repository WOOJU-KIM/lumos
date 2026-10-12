import sys
import subprocess
import shutil
import re
from pathlib import Path

def update_config(symbol_long, symbol_short, macro_symbol, use_60m, trigger, drop):
    with open('config.py', 'r', encoding='utf-8') as f:
        c = f.read()
    c = re.sub(r'BASE_ASSET_LONG\s*=\s*".*"', f'BASE_ASSET_LONG="{symbol_long}"', c)
    c = re.sub(r'BASE_ASSET_SHORT\s*=\s*".*"', f'BASE_ASSET_SHORT="{symbol_short}"', c)
    c = re.sub(r'MACRO_TREND_SYMBOL\s*=\s*".*"', f'MACRO_TREND_SYMBOL="{macro_symbol}"', c)
    c = re.sub(r'USE_60M_TREND_FILTER\s*=\s*(True|False)', f'USE_60M_TREND_FILTER = {use_60m}', c)
    c = re.sub(r'TRAILING_TRIGGER_PCT\s*=\s*[0-9.]+', f'TRAILING_TRIGGER_PCT = {trigger}', c)
    c = re.sub(r'TRAILING_DROP_PCT\s*=\s*[0-9.]+', f'TRAILING_DROP_PCT = {drop}', c)
    with open('config.py', 'w', encoding='utf-8') as f:
        f.write(c)

print("Running TQQQ with 60m OFF and 0.8/0.4 trailing...")
update_config("TQQQ", "SQQQ", "QQQ", "False", "0.008", "0.004")
subprocess.run([sys.executable, "scripts/run_weekly_rolling_wfa.py"])
shutil.copy("data/backtest_weekly_rolling_wfa_summary.json", "data/summary_tqqq_0.8.json")

# Restore original SOXL Option A setting so live trading isn't broken
# SOXL, 60m OFF, 1.0 / 0.5
update_config("SOXL", "SOXS", "SOXX", "False", "0.010", "0.005")
print("Tests complete and SOXL config restored.")
