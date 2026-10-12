import urllib.request
import json
import sqlite3
import os
import datetime

db_path = os.path.join('c:\\', 'Users', 'chabo', 'OneDrive', '바탕 화면', 'lumos', 'data', 'market_data.db')
conn = sqlite3.connect(db_path)
c = conn.cursor()
c.execute("SELECT open, high, low, close FROM market_candles WHERE symbol='SQQQ' AND timeframe='15m' AND datetime='2026-09-30 09:30:00'")
alpaca_row = c.fetchone()
print(f"Alpaca 22:30 KST (09:30 NYT):")
print(f"Open: {alpaca_row[0]:.4f}, High: {alpaca_row[1]:.4f}, Low: {alpaca_row[2]:.4f}, Close: {alpaca_row[3]:.4f}")

url = "https://query1.finance.yahoo.com/v8/finance/chart/SQQQ?interval=15m&range=2d"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
resp = urllib.request.urlopen(req).read()
data = json.loads(resp)
timestamps = data['chart']['result'][0]['timestamp']
quotes = data['chart']['result'][0]['indicators']['quote'][0]
print(f"\nYahoo Finance 22:30 KST (09:30 NYT):")
for t, o, h, l, c_val in zip(timestamps, quotes['open'], quotes['high'], quotes['low'], quotes['close']):
    dt = datetime.datetime.fromtimestamp(t)
    if '22:30' in str(dt) and '09-30' in str(dt):
        print(f"Open: {o:.4f}, High: {h:.4f}, Low: {l:.4f}, Close: {c_val:.4f}")
