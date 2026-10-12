import sqlite3
db=r'C:\Users\chabo\OneDrive\바탕 화면\lumos\data\live_experience.db'
conn=sqlite3.connect(db)
c=conn.cursor()
print(c.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='daily_portfolio_history'").fetchone()[0])
print("\nCurrent data:")
for row in c.execute("SELECT * FROM daily_portfolio_history"):
    print(row)
conn.close()
