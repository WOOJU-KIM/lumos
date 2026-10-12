import sqlite3
conn = sqlite3.connect('c:/Users/chabo/OneDrive/?? ??/lumos/data/market_data.db')
c = conn.cursor()
for row in c.execute("SELECT datetime, open, high, low, close FROM market_candles WHERE symbol='SQQQ' AND datetime LIKE '2026-09-30 %' ORDER BY datetime ASC"):
    if '09:30' in row[0] or '11:45' in row[0] or '14:00' in row[0]:
        print(row)
