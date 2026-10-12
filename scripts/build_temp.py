import os

with open('scripts/run_weekly_rolling_wfa.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open('scripts/temp_run_wfa_oct_1_2.py', 'w', encoding='utf-8') as f:
    for line in lines:
        if 'for w_idx in range(test_start_idx, len(unique_weeks)):' in line:
            f.write('    for w_idx in range(len(unique_weeks) - 1, len(unique_weeks)):\n')
        elif 'df_trades = pd.DataFrame(trades)' in line:
            f.write('    df_trades = pd.DataFrame(trades)\n')
            f.write('    print("\\n=== Trades for the target week ===")\n')
            f.write('    for t in trades:\n')
            f.write('        print(f"Entry: {t[\'entry_dt\']} | Exit: {t[\'exit_dt\']} | Sym: {t[\'symbol\']} | Entry Px: {t[\'entry_px\']:.2f} | Exit Px: {t[\'exit_px\']:.2f} | Ret: {t[\'net_ret_pct\']}% | Reason: {t[\'exit_reason\']}")\n')
            f.write('    print("====================================\\n")\n')
        else:
            f.write(line)
