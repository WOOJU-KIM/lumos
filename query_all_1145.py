# -*- coding: utf-8 -*-
import sqlite3
import os
path = os.path.join('c:\\', 'Users', 'chabo', 'OneDrive', '바탕 화면', 'lumos', 'data', 'market_data.db')
conn = sqlite3.connect(path)
c = conn.cursor()
for row in c.execute("SELECT symbol, timeframe, datetime, open, high, low, close FROM market_candles WHERE datetime LIKE '2026-09-30 11:45%' AND timeframe='15m'"):
    print(row)