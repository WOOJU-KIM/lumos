import sqlite3
import pandas as pd

try:
    conn = sqlite3.connect('shadow_trades.db')
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print('Tables in shadow_trades.db:', tables)
    if tables:
        df = pd.read_sql_query(f"SELECT * FROM {tables[0]} LIMIT 5", conn)
        print(df.to_string())
except Exception as e:
    print(e)
