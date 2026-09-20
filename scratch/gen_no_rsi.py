import sys
with open('scripts/run_wfa_live_identical_v2_fixed.py', 'r', encoding='utf-8') as f:
    code = f.read()

code = code.replace(
    "if dir_gbdt == \"LONG_TQQQ\" and rsi_14 > config.RSI_OVERBOUGHT_THRESHOLD: continue",
    "if dir_gbdt == \"LONG_TQQQ\" and rsi_14 > 100: continue"
).replace(
    "if dir_gbdt == \"SHORT_SQQQ\" and rsi_14 < config.RSI_OVERSOLD_THRESHOLD: continue",
    "if dir_gbdt == \"SHORT_SQQQ\" and rsi_14 < 0: continue"
).replace(
    "output_path = DATA_DIR / \"wfa_live_identical_trades.csv\"",
    "output_path = DATA_DIR / \"wfa_no_rsi_trades.csv\""
)

with open('scripts/run_no_rsi_test.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("No-RSI script generated successfully.")
