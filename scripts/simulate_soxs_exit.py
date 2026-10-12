import sqlite3
import config
import pandas as pd

conn = sqlite3.connect('data/market_data.db')
df = pd.read_sql_query("""
    SELECT datetime, open, high, low, close, volume 
    FROM market_candles 
    WHERE symbol=config.BASE_ASSET_SHORT AND timeframe='5m' AND datetime >= '2026-08-21 10:00:00'
    ORDER BY datetime ASC
""", conn)
conn.close()

buy_px = 47.67
qty = 1972
target_tp = 49.31  # +3.5%
target_sl = round(buy_px * 0.975, 2) # -2.5% -> 46.48 (or -2.0% -> 46.72)
sl_20 = round(buy_px * 0.98, 2) # 46.72
sl_25 = round(buy_px * 0.975, 2) # 46.48

print(f"진입 시점: 2026-08-21 10:17:06 NYT (23:17:06 KST)")
print(f"진입 가격: ${buy_px:.2f} | 수량: {qty:,}주 | 투자원금: ${buy_px * qty:,.2f}")
print(f"목표 익절가 (+3.5%): ${target_tp:.2f}")
print(f"칼손절선 (-2.0%): ${sl_20:.2f} | (-2.5% Cap): ${sl_25:.2f}")
print(f"90분 타임스탑 시점: 2026-08-21 11:47:06 NYT (00:47:06 KST)\n")

print("--- 2026-08-21 10:15 NYT 이후 5분봉 궤적 ---")
for idx, row in df.iterrows():
    dt = row['datetime']
    o, h, l, c = row['open'], row['high'], row['low'], row['close']
    h_ret = (h - buy_px) / buy_px * 100
    l_ret = (l - buy_px) / buy_px * 100
    c_ret = (c - buy_px) / buy_px * 100
    print(f"[{dt}] O:{o:.2f} H:{h:.2f} L:{l:.2f} C:{c:.2f} | 변동률 High:{h_ret:+.2f}% Low:{l_ret:+.2f}% Close:{c_ret:+.2f}%")
