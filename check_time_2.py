import sqlite3
conn = sqlite3.connect('data/market_data.db')
print('15m')
for row in conn.execute("SELECT datetime, close FROM market_candles WHERE symbol='TQQQ' AND timeframe='15m' ORDER BY datetime DESC LIMIT 5"):
    print(row)
print('5m')
for row in conn.execute("SELECT datetime, close FROM market_candles WHERE symbol='TQQQ' AND timeframe='5m' ORDER BY datetime DESC LIMIT 10"):
    print(row)
