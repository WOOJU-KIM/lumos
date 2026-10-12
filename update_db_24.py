import sqlite3
conn = sqlite3.connect(r'c:\Users\chabo\OneDrive\바탕 화면\lumos\data\live_experience.db')
cursor = conn.cursor()
cursor.execute("UPDATE daily_portfolio_history SET balance_usd = 63062.0 WHERE date = '2026-09-24'")
prev_bal = 63062.0
curr_bal = 59547.0
ret_pct = round(((curr_bal - prev_bal) / prev_bal) * 100.0, 2)
cursor.execute("UPDATE daily_portfolio_history SET return_pct_usd = ? WHERE date = '2026-09-25'", (ret_pct,))
conn.commit()
import pandas as pd
df = pd.read_sql_query('SELECT date, balance, return_pct, balance_usd, return_pct_usd FROM daily_portfolio_history ORDER BY date DESC LIMIT 5', conn)
print(df.to_string())
conn.close()
