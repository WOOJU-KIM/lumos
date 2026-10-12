import urllib.request
import json
import sqlite3
import os
import datetime

# 1. Fetch Alpaca data from DB
db_path = os.path.join('c:\\', 'Users', 'chabo', 'OneDrive', '바탕 화면', 'lumos', 'data', 'market_data.db')
conn = sqlite3.connect(db_path)
c = conn.cursor()
c.execute("SELECT open, high, low, close FROM market_candles WHERE symbol='SQQQ' AND timeframe='15m' AND datetime='2026-09-30 11:45:00'")
alpaca_row = c.fetchone()
print(f"Alpaca (Lumos DB) 00:45 KST (11:45 NYT):")
if alpaca_row:
    print(f"Open: {alpaca_row[0]:.4f}, High: {alpaca_row[1]:.4f}, Low: {alpaca_row[2]:.4f}, Close: {alpaca_row[3]:.4f}")
else:
    print("Not found in DB")

# 2. Fetch Yahoo Finance data
url = "https://query1.finance.yahoo.com/v8/finance/chart/SQQQ?interval=15m&range=2d"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    resp = urllib.request.urlopen(req).read()
    data = json.loads(resp)
    timestamps = data['chart']['result'][0]['timestamp']
    quotes = data['chart']['result'][0]['indicators']['quote'][0]
    print(f"\nYahoo Finance 00:45 KST (11:45 NYT):")
    found = False
    for t, o, h, l, c_val in zip(timestamps, quotes['open'], quotes['high'], quotes['low'], quotes['close']):
        dt = datetime.datetime.fromtimestamp(t)
        # We look for NYT 11:45, but the timestamp is local time.
        # So we can just look for the time when the Alpaca candle matches, which is 00:45 KST (if my environment is KST)
        if '00:45' in str(dt) and '10-01' in str(dt):
            print(f"Open: {o:.4f}, High: {h:.4f}, Low: {l:.4f}, Close: {c_val:.4f}")
            found = True
    if not found:
        print("Not found in Yahoo data for this timestamp")
except Exception as e:
    print("Yahoo fetch failed:", e)

