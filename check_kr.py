import sqlite3
conn = sqlite3.connect(r'c:\Users\chabo\OneDrive\바탕 화면\lumos\data\market_data.db')
print([row[0] for row in conn.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall()])
