import config
import pandas as pd
import sqlite3

csv_path = 'data/live_trades.csv'
db_path = 'data/live_experience.db'

# CSV 삭제
try:
    df = pd.read_csv(csv_path)
    before_len = len(df)
    df = df[df['symbol'] != config.BASE_ASSET_LONG]
    after_len = len(df)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"CSV에서 {before_len - after_len}건 삭제 완료.")
except Exception as e:
    print(f"CSV 오류: {e}")

# DB 삭제
try:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("DELETE FROM live_trades WHERE symbol=config.BASE_ASSET_LONG")
    conn.commit()
    deleted_rows = c.rowcount
    conn.close()
    print(f"DB에서 {deleted_rows}건 삭제 완료.")
except Exception as e:
    print(f"DB 오류: {e}")
