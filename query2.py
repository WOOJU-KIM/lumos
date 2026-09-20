import sqlite3
from config import DATA_DIR
conn = sqlite3.connect(DATA_DIR / 'market_data.db')
print(conn.execute("SELECT MIN(datetime), MAX(datetime) FROM market_candles WHERE symbol='TQQQ'").fetchone())
print(conn.execute("SELECT MIN(datetime), MAX(datetime) FROM market_candles WHERE symbol='SOXL'").fetchone())
