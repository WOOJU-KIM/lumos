import re

with open('scripts/temp_run_wfa_10yr.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Time Travel Bug (Future Leakage)
content = re.sub(
    r"future_5m = future_5m\[future_5m\.index > curr_dt_ts\]",
    "import pandas as pd\n                    future_5m = future_5m[future_5m.index >= curr_dt_ts + pd.Timedelta(minutes=15)]",
    content
)

# Fix 2: TP Cap Bypass Bug (Trailing SL exceeding Max TP)
old_sl_logic = """                        if cur_low <= current_sl_px:
                            exit_price = current_sl_px
                            exit_reason = "TRAILING_SL" if current_sl_px > sl_px else "STOP_LOSS\""""

new_sl_logic = """                        if cur_low <= current_sl_px:
                            if current_sl_px > tp_px:
                                exit_price = tp_px
                                exit_reason = "MAX_TP"
                            else:
                                exit_price = current_sl_px
                                exit_reason = "TRAILING_SL" if current_sl_px > sl_px else "STOP_LOSS\""""

content = content.replace(old_sl_logic, new_sl_logic)

with open('scripts/temp_run_wfa_10yr.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Sandbox patched successfully with bug fixes.")
