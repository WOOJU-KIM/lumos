import pandas as pd
df = pd.read_csv(r'c:\Users\chabo\OneDrive\바탕 화면\lumos\data\live_trades.csv', encoding='utf-8')
df['us_session_date'] = (pd.to_datetime(df['entry_time']) - pd.Timedelta(hours=12)).dt.date
print(df[df['us_session_date'].astype(str) == '2026-09-23'][['trade_id', 'symbol', 'quantity', 'pnl_usd', 'pnl_pct', 'actual_entry_price', 'actual_exit_price']].to_string())
