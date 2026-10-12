import pandas as pd
df = pd.read_csv(r'c:\Users\chabo\OneDrive\바탕 화면\lumos\data\live_trades.csv', encoding='utf-8')
df['exit_date'] = pd.to_datetime(df['exit_time']).dt.date
daily = df.groupby('exit_date').agg(Trades=('trade_id', 'count'), Win_Rate=('pnl_usd', lambda x: (x>0).mean()*100), PnL_USD=('pnl_usd', 'sum')).reset_index()
print(daily.to_string())
