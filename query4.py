# -*- coding: utf-8 -*-
import sqlite3
import os
path = os.path.join('c:\\', 'Users', 'chabo', 'OneDrive', '바탕 화면', 'lumos', 'data', 'market_data.db')
conn = sqlite3.connect(path)
c = conn.cursor()
for row in c.execute("SELECT datetime, open, high, low, close FROM market_candles WHERE symbol='SQQQ' AND datetime LIKE '2026-09-30 %' ORDER BY datetime ASC"):
    if '09:30' in row[0] or '11:45' in row[0] or '14:00' in row[0]:
        print(row)