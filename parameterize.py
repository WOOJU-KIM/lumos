import sys
import re

with open('scripts/run_weekly_rolling_wfa.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add arguments support
arg_parsing = """import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--long', default='TQQQ')
parser.add_argument('--short', default='SQQQ')
parser.add_argument('--filter', default='QQQ')
args = parser.parse_args()

SYM_LONG = args.long
SYM_SHORT = args.short
SYM_FILTER = args.filter

"""
content = content.replace("def run_weekly_rolling_walk_forward():", arg_parsing + "def run_weekly_rolling_walk_forward():")

# 2. Replace hardcoded TQQQ/SQQQ/QQQ loading
load_old = """    tqqq_15m = lake.load_candles("TQQQ", "15m")
    sqqq_15m = lake.load_candles("SQQQ", "15m")
    qqq_60m = lake.load_candles("QQQ", "60m")"""
load_new = """    tqqq_15m = lake.load_candles(SYM_LONG, "15m")
    sqqq_15m = lake.load_candles(SYM_SHORT, "15m")
    qqq_60m = lake.load_candles(SYM_FILTER, "60m")"""
content = content.replace(load_old, load_new)

# 3. Replace active_pos
content = content.replace("'sym': 'TQQQ'", "'sym': SYM_LONG")
content = content.replace("'sym': 'SQQQ'", "'sym': SYM_SHORT")

# 4. Replace string matching for close prices
content = content.replace("cur_tqqq_close if sym == 'TQQQ'", "cur_tqqq_close if sym == SYM_LONG")
content = content.replace("cur_tqqq_high if sym == 'TQQQ'", "cur_tqqq_high if sym == SYM_LONG")
content = content.replace("cur_tqqq_low if sym == 'TQQQ'", "cur_tqqq_low if sym == SYM_LONG")

# 5. Output file name based on symbols
content = content.replace('"data/backtest_weekly_rolling_wfa_summary.json"', 'f"data/backtest_wfa_summary_{SYM_LONG}.json"')
content = content.replace('"data/backtest_weekly_rolling_wfa_trades.csv"', 'f"data/backtest_wfa_trades_{SYM_LONG}.csv"')

with open('scripts/run_weekly_rolling_wfa_multi.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("CREATED run_weekly_rolling_wfa_multi.py")
