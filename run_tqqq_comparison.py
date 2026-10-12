import sys
import subprocess
import json
import re
import shutil
from pathlib import Path

def update_config(symbol_long, symbol_short, macro_symbol, use_60m):
    with open('config.py', 'r', encoding='utf-8') as f:
        c = f.read()
    c = re.sub(r'BASE_ASSET_LONG\s*=\s*".*"', f'BASE_ASSET_LONG="{symbol_long}"', c)
    c = re.sub(r'BASE_ASSET_SHORT\s*=\s*".*"', f'BASE_ASSET_SHORT="{symbol_short}"', c)
    c = re.sub(r'MACRO_TREND_SYMBOL\s*=\s*".*"', f'MACRO_TREND_SYMBOL="{macro_symbol}"', c)
    c = re.sub(r'USE_60M_TREND_FILTER\s*=\s*(True|False)', f'USE_60M_TREND_FILTER = {use_60m}', c)
    with open('config.py', 'w', encoding='utf-8') as f:
        f.write(c)

print("Running TQQQ with 60m OFF...")
update_config("TQQQ", "SQQQ", "QQQ", "False")
subprocess.run([sys.executable, "scripts/run_weekly_rolling_wfa.py"])
shutil.copy("data/backtest_weekly_rolling_wfa_summary.json", "data/summary_tqqq_60m_off.json")

print("Running TQQQ with 60m ON...")
update_config("TQQQ", "SQQQ", "QQQ", "True")
subprocess.run([sys.executable, "scripts/run_weekly_rolling_wfa.py"])
shutil.copy("data/backtest_weekly_rolling_wfa_summary.json", "data/summary_tqqq_60m_on.json")

# Restore original SOXL Option A setting so live trading isn't broken
update_config("SOXL", "SOXS", "SOXX", "False")
print("Tests complete and SOXL config restored.")
