import sqlite3
conn = sqlite3.connect('data/market_data.db')
schema = conn.execute("PRAGMA table_info(market_candles);").fetchall()
print(schema)
