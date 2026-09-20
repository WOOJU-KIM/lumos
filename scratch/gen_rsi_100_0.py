with open('scripts/run_wfa_live_identical_v2_fixed.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Replace config.RSI_OVERBOUGHT_THRESHOLD with 100
code = code.replace(
    "if dir_gbdt == \"LONG_TQQQ\" and rsi_14 > config.RSI_OVERBOUGHT_THRESHOLD: continue",
    "if dir_gbdt == \"LONG_TQQQ\" and rsi_14 > 100: continue"
)

# Replace config.RSI_OVERSOLD_THRESHOLD with 0
code = code.replace(
    "if dir_gbdt == \"SHORT_SQQQ\" and rsi_14 < config.RSI_OVERSOLD_THRESHOLD: continue",
    "if dir_gbdt == \"SHORT_SQQQ\" and rsi_14 < 0: continue"
)

# Change output path
code = code.replace(
    "output_path = DATA_DIR / \"wfa_live_identical_trades.csv\"",
    "output_path = DATA_DIR / \"wfa_trades_rsi_100_0.csv\""
)

with open('scripts/run_wfa_rsi_100_0_test.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("Generated scripts/run_wfa_rsi_100_0_test.py successfully.")
