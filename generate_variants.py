with open('scripts/run_weekly_rolling_wfa.py', 'r', encoding='utf-8') as f:
    base = f.read()

# Generate SOXL
soxl = base.replace("TQQQ", "SOXL").replace("tqqq", "soxl")
soxl = soxl.replace("SQQQ", "SOXS").replace("sqqq", "soxs")
soxl = soxl.replace("QQQ", "SOXX").replace("qqq", "soxx")
soxl = soxl.replace("backtest_weekly_rolling_wfa_summary", "backtest_soxl_summary")
soxl = soxl.replace("backtest_weekly_rolling_wfa_trades", "backtest_soxl_trades")

with open('scripts/run_soxl_wfa.py', 'w', encoding='utf-8') as f:
    f.write(soxl)

# Generate UPRO
upro = base.replace("TQQQ", "UPRO").replace("tqqq", "upro")
upro = upro.replace("SQQQ", "SPXU").replace("sqqq", "spxu")
upro = upro.replace("QQQ", "SPY").replace("qqq", "spy")
upro = upro.replace("backtest_weekly_rolling_wfa_summary", "backtest_upro_summary")
upro = upro.replace("backtest_weekly_rolling_wfa_trades", "backtest_upro_trades")

with open('scripts/run_upro_wfa.py', 'w', encoding='utf-8') as f:
    f.write(upro)

print("Generated variants.")
