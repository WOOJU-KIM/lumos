import sys
import re

long_sym = sys.argv[1]
short_sym = sys.argv[2]
filter_sym = sys.argv[3]

with open('scripts/run_weekly_rolling_wfa.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the data loading specifically
content = re.sub(r'lake\.load_candles\("TQQQ", "15m"\)', f'lake.load_candles("{long_sym}", "15m")', content)
content = re.sub(r'lake\.load_candles\("SQQQ", "15m"\)', f'lake.load_candles("{short_sym}", "15m")', content)
content = re.sub(r'lake\.load_candles\("QQQ", "60m"\)', f'lake.load_candles("{filter_sym}", "60m")', content)

# Output filenames
content = content.replace('backtest_weekly_rolling_wfa_summary.json', f'backtest_summary_{long_sym}.json')
content = content.replace('backtest_weekly_rolling_wfa_trades.csv', f'backtest_trades_{long_sym}.csv')

# String literals
content = content.replace("'TQQQ'", f"'{long_sym}'")
content = content.replace("'SQQQ'", f"'{short_sym}'")
content = content.replace('"TQQQ"', f'"{long_sym}"')
content = content.replace('"SQQQ"', f'"{short_sym}"')
content = content.replace('LONG_TQQQ', f'LONG_{long_sym}')
content = content.replace('SHORT_SQQQ', f'SHORT_{short_sym}')

content = content.replace("TQQQ/SQQQ", f"{long_sym}/{short_sym}")
content = content.replace("TQQQ:", f"{long_sym}:")
content = content.replace("SQQQ:", f"{short_sym}:")
content = content.replace("tqqq_cnt", f"{long_sym.lower()}_cnt")
content = content.replace("sqqq_cnt", f"{short_sym.lower()}_cnt")
content = content.replace("tqqq_trades", f"{long_sym.lower()}_trades")
content = content.replace("sqqq_trades", f"{short_sym.lower()}_trades")

out_file = f'scripts/run_temp_{long_sym}.py'
with open(out_file, 'w', encoding='utf-8') as f:
    f.write(content)
print(f"Created {out_file}")
