import json

with open('scratch/temp_run_rolling_oos_db_backtest.py', 'a', encoding='utf-8') as f:
    f.write('''
    # Markdown Table Output
    print("\\n### 📊 백테스트 결과 요약 (전체 기간 롤링 OOS)")
    print("| 종목 | 총 수익률 | 주별 매매횟수 | 연평균수익률(CAGR) | MDD |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    
    total_ret = (current_capital / initial_capital - 1.0) * 100
    weeks = len(trades) / (len(trades) / max(1, len(monthly_summary) * 4)) if len(trades) > 0 else 0
    weekly_trades = len(trades) / (len(monthly_summary) * 4) if len(monthly_summary) > 0 else 0
    cagr = ((current_capital / initial_capital) ** (1 / (len(monthly_summary) / 12)) - 1.0) * 100 if len(monthly_summary) > 0 else 0

    print(f"| TQQQ/SQQQ | {total_ret:+.2f}% | {weekly_trades:.1f}회 | {cagr:+.2f}% | {max_drawdown_pct:.2f}% |")
''')
