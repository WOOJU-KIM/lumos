import pandas as pd, sqlite3
conn = sqlite3.connect(r'c:\Users\chabo\OneDrive\바탕 화면\lumos\data\live_experience.db')
df = pd.read_sql_query('SELECT * FROM live_system_errors ORDER BY timestamp DESC LIMIT 5', conn)
df.to_csv('errors.csv', index=False, encoding='utf-8-sig')
