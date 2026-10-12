# -*- coding: utf-8 -*-
import sqlite3
import os
path = os.path.join('c:\\', 'Users', 'chabo', 'OneDrive', '바탕 화면', 'lumos', 'data', 'market_data.db')
conn = sqlite3.connect(path)
c = conn.cursor()
for row in c.execute("SELECT timeframe, datetime, open, high, low, close FROM market_candles WHERE symbol='SQQQ' AND datetime LIKE '2026-09-30 %' AND (high BETWEEN 33.5 AND 33.6 OR low BETWEEN 33.5 AND 33.6) ORDER BY datetime ASC"):
    print(row)