import sys
with open('scripts/run_regime_filter_test.py', 'r', encoding='utf-8') as f:
    code = f.read()

code = code.replace(
    "soxx_60m['ema20'] = soxx_60m['Close'].ewm(span=20, adjust=False).mean()",
    "soxx_60m['ema20'] = soxx_60m['Close'].ewm(span=FAST_EMA, adjust=False).mean()"
).replace(
    "soxx_60m['ema50'] = soxx_60m['Close'].ewm(span=50, adjust=False).mean()",
    "soxx_60m['ema50'] = soxx_60m['Close'].ewm(span=SLOW_EMA, adjust=False).mean()"
).replace(
    "if len(past_soxx) < 50: continue",
    "if len(past_soxx) < SLOW_EMA: continue"
).replace(
    "output_path = DATA_DIR / \"wfa_regime_test_trades.csv\"",
    "output_path = DATA_DIR / f\"wfa_regime_test_{FAST_EMA}_{SLOW_EMA}.csv\""
)

header = '''
import sys
FAST_EMA = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SLOW_EMA = int(sys.argv[2]) if len(sys.argv) > 2 else 50
'''
code = header + code

with open('scripts/run_doe_regime.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("DOE script generated successfully.")
