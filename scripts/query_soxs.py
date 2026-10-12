import sqlite3
import config
import pandas as pd

conn = sqlite3.connect('data/market_data.db')
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(market_candles)")
cols = [c[1] for c in cursor.fetchall()]
print("Columns:", cols)

# Query SQQQ
symbol_col = "symbol" if "symbol" in cols else "ticker"
df = pd.read_sql_query(f"SELECT * FROM market_candles WHERE {symbol_col}=config.BASE_ASSET_SHORT ORDER BY datetime DESC LIMIT 100", conn)
conn.close()

print("\n--- SQQQ Candles ---")
print(df.head(40).to_string())
