import pandas as pd, sqlite3
conn = sqlite3.connect(r'c:\Users\chabo\OneDrive\바탕 화면\lumos\data\live_experience.db')
df = pd.read_sql_query('SELECT * FROM daily_portfolio_history ORDER BY date DESC LIMIT 10', conn)
print(df.to_string())
