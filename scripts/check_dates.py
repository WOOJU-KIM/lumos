import sqlite3

conn = sqlite3.connect('data/market_data.db')
c = conn.cursor()
c.execute("SELECT MIN(datetime), MAX(datetime) FROM market_candles WHERE symbol='SOXL' AND timeframe='15m'")
print(c.fetchone())
conn.close()
