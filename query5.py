# -*- coding: utf-8 -*-
import sqlite3
import os
path = os.path.join('c:\\', 'Users', 'chabo', 'OneDrive', '바탕 화면', 'lumos', 'data', 'market_data.db')
conn = sqlite3.connect(path)
c = conn.cursor()
for row in c.execute("SELECT datetime, open, high, low, close FROM daily_data WHERE symbol='SQQQ' AND datetime LIKE '2026-09-30 09:30%'"):
    print(row)