import sqlite3
import pandas as pd

conn = sqlite3.connect(r'c:\Users\chabo\OneDrive\바탕 화면\lumos\data\live_experience.db')
df = pd.read_sql_query('SELECT * FROM live_trades', conn)

df['entry_time'] = pd.to_datetime(df['entry_time'])
df['us_session_date'] = (df['entry_time'] - pd.Timedelta(hours=12)).dt.date.astype(str)

# Filter glitch
df = df[df['actual_entry_price'] > 0]

daily = df.groupby('us_session_date').agg(
    total_trades=('trade_id', 'count'),
    win_trades=('pnl_usd', lambda x: (x > 0).sum()),
    total_pnl_usd=('pnl_usd', 'sum')
).reset_index()

daily['win_rate_pct'] = round((daily['win_trades'] / daily['total_trades']) * 100, 1)
daily['total_pnl_usd'] = round(daily['total_pnl_usd'], 2)

cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS daily_trade_pnl (
    us_session_date TEXT PRIMARY KEY,
    total_trades INTEGER,
    win_trades INTEGER,
    win_rate_pct REAL,
    total_pnl_usd REAL
)
''')

for _, row in daily.iterrows():
    cursor.execute('''
    INSERT OR REPLACE INTO daily_trade_pnl 
    (us_session_date, total_trades, win_trades, win_rate_pct, total_pnl_usd) 
    VALUES (?, ?, ?, ?, ?)
    ''', (row['us_session_date'], row['total_trades'], row['win_trades'], row['win_rate_pct'], row['total_pnl_usd']))

conn.commit()

final_df = pd.read_sql_query('SELECT * FROM daily_trade_pnl ORDER BY us_session_date DESC LIMIT 10', conn)
print(final_df.to_string())

conn.close()
