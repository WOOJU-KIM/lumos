import pandas as pd
import datetime

with open('run_v5_rolling_backtest.py', 'r', encoding='utf-8') as f:
    code = f.read()

code = code.replace(
'''    min_date = master['Datetime'].min()
    max_date = master['Datetime'].max()
    test_start = min_date + timedelta(days=730)
    while test_start.weekday() != 0: test_start += timedelta(days=1)
    test_start = test_start.replace(hour=0, minute=0, second=0)''',
'''    min_date = master['Datetime'].min()
    max_date = pd.to_datetime("2026-09-19 00:00:00")
    test_start = pd.to_datetime("2026-09-14 00:00:00")'''
)

save_code = '''
    report_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_fast_backtest_report.md"
    report_path = report_dir / report_filename
    with open(report_path, 'w', encoding='utf-8') as f:
         f.write(f"# Lumos V5 Fast Vectorized Backtest\\n**기간**: {years_elapsed:.1f}년\\n**수익률**: {total_return_pct:.2f}%\\n**MDD**: {max_drawdown:.2f}%\\n**거래횟수**: {total_trades}건\\n")
    print(f"   ✅ 리포트 저장 완료: {report_path}")

    import pandas as pd
    if trades:
        df_trades = pd.DataFrame(trades)
        df_trades.rename(columns={'date': 'entry_dt', 'capital_after': 'capital'}, inplace=True)
        df_trades.to_csv('data/wfa_tqqq_trades.csv', index=False, encoding='utf-8-sig')
        # UPRO trades empty placeholder for chart
        pd.DataFrame(columns=['entry_dt', 'capital']).to_csv('data/wfa_upro_trades.csv', index=False)
        print("✅ Saved trades to data/wfa_tqqq_trades.csv")
'''

code = code.replace(
'''    report_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_fast_backtest_report.md"
    report_path = report_dir / report_filename
    with open(report_path, 'w', encoding='utf-8') as f:
         f.write(f"# Lumos V5 Fast Vectorized Backtest\\n**기간**: {years_elapsed:.1f}년\\n**수익률**: {total_return_pct:.2f}%\\n**MDD**: {max_drawdown:.2f}%\\n**거래횟수**: {total_trades}건\\n")
    print(f"   ✅ 리포트 저장 완료: {report_path}")''',
save_code
)

with open('temp_run_custom_week.py', 'w', encoding='utf-8') as f:
    f.write(code)
