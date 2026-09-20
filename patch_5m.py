import pandas as pd
with open('scripts/run_5m_precision_backtest.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Filter train loop
code = code.replace(
'''        for w_idx in range(ROLLING_WINDOW_WEEKS, len(unique_weeks)):
            cur_test_week = unique_weeks[w_idx]''',
'''        for w_idx in range(ROLLING_WINDOW_WEEKS, len(unique_weeks)):
            cur_test_week = unique_weeks[w_idx]
            if not ('2026-36' <= cur_test_week <= '2026-40'): continue'''
)

# 2. Filter dates loop
code = code.replace(
'''    unique_dates = sorted(test_df['date_str'].unique())

    for d_str in unique_dates:''',
'''    unique_dates = sorted(test_df['date_str'].unique())

    for d_str in unique_dates:
        if d_str < '2026-08-31' or d_str > '2026-10-02': continue'''
)

# 3. Save trades CSV
csv_save_code = '''    out_file = PROJECT_ROOT / "data" / f"backtest_5m_precision_{model_train_mode}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(summary_data), f, indent=2, ensure_ascii=False)
    print(f"\\n💾 [5분봉 정밀 백테스트 결과 저장 완료] {out_file}")

    # Export for plotting
    df_trades = pd.DataFrame(trades)
    if not df_trades.empty:
        df_trades.rename(columns={'entry_dt': 'entry_dt', 'ending_capital': 'capital', 'exit_time': 'exit_time'}, inplace=True)
        # We need entry_time and exit_time string format HH:MM
        df_trades['entry_time'] = df_trades['entry_dt'].str[11:16]
        df_trades['exit_time'] = df_trades['exit_dt'].str[11:16]
        df_trades['entry_dt'] = df_trades['entry_dt'].str[:10]
        df_trades['direction'] = df_trades['symbol'].apply(lambda x: 'LONG_TQQQ' if x == 'TQQQ' else 'SHORT_SQQQ')
        df_trades['return_pct'] = df_trades['net_ret_pct']
        df_trades['pnl_usd'] = df_trades['pnl_krw']
        df_trades['entry_price'] = df_trades['entry_px']
        df_trades['exit_price'] = df_trades['exit_px']
        
        df_trades[['entry_dt', 'symbol', 'direction', 'entry_time', 'entry_price', 'exit_time', 'exit_price', 'return_pct', 'pnl_usd', 'capital', 'exit_reason']].to_csv('data/wfa_tqqq_trades.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(columns=['entry_dt', 'capital']).to_csv('data/wfa_upro_trades.csv', index=False)
        print("✅ Saved CSV for plotting.")
    else:
        pd.DataFrame(columns=['entry_dt', 'capital']).to_csv('data/wfa_tqqq_trades.csv', index=False)
        pd.DataFrame(columns=['entry_dt', 'capital']).to_csv('data/wfa_upro_trades.csv', index=False)
'''

code = code.replace(
'''    out_file = PROJECT_ROOT / "data" / f"backtest_5m_precision_{model_train_mode}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(summary_data), f, indent=2, ensure_ascii=False)
    print(f"\\n💾 [5분봉 정밀 백테스트 결과 저장 완료] {out_file}")''',
csv_save_code
)

with open('temp_run_sep_5m.py', 'w', encoding='utf-8') as f:
    f.write(code)
