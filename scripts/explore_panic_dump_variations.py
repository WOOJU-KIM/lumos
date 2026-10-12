import sqlite3
import sys
import config
import pandas as pd
import numpy as np

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

def explore_variations():
    conn = sqlite3.connect('data/market_data.db')
    
    # Load 5m TQQQ and SQQQ
    df_l = pd.read_sql_query("SELECT datetime, open, high, low, close, volume FROM market_candles WHERE symbol=config.BASE_ASSET_LONG AND timeframe='5m' ORDER BY datetime ASC", conn)
    df_s = pd.read_sql_query("SELECT datetime, open, high, low, close, volume FROM market_candles WHERE symbol=config.BASE_ASSET_SHORT AND timeframe='5m' ORDER BY datetime ASC", conn)
    
    df_l['datetime'] = pd.to_datetime(df_l['datetime'])
    df_l['date'] = df_l['datetime'].dt.strftime('%Y-%m-%d')
    df_l['time'] = df_l['datetime'].dt.strftime('%H:%M')

    df_s['datetime'] = pd.to_datetime(df_s['datetime'])
    df_s['date'] = df_s['datetime'].dt.strftime('%Y-%m-%d')
    df_s['time'] = df_s['datetime'].dt.strftime('%H:%M')

    days = sorted(df_l['date'].unique())
    
    # Test different entry times: 14:45, 15:00, 15:15
    for entry_t in ['14:45', '15:00', '15:15']:
        trades = []
        for d in days:
            sl = df_l[df_l['date'] == d].sort_values('time')
            ss = df_s[df_s['date'] == d].sort_values('time')
            if len(sl) < 50: continue

            # Day open
            d_open = sl[sl['time'] == '09:30'].iloc[0]['open']
            
            # Prior 1 hour box (e.g. entry_t - 60min)
            t_min = (pd.to_datetime(f"2026-01-01 {entry_t}:00") - pd.Timedelta(minutes=60)).strftime('%H:%M')
            box = sl[(sl['time'] >= t_min) & (sl['time'] < entry_t)]
            if len(box) < 6: continue
            box_low = box['low'].min()

            bar_entry_l = sl[sl['time'] == entry_t]
            bar_entry_s = ss[ss['time'] == entry_t]
            if bar_entry_l.empty or bar_entry_s.empty: continue

            c_l = bar_entry_l.iloc[0]['close']
            c_s = bar_entry_s.iloc[0]['close']

            day_ret = (c_l - d_open) / d_open * 100

            # Condition: broke below box low
            if c_l < box_low:
                entry_px = c_s + 0.03
                post_s = ss[(ss['time'] > entry_t) & (ss['time'] <= '15:50')]
                if post_s.empty: continue
                exit_px = post_s.iloc[-1]['close'] - 0.03
                net_ret = (exit_px - entry_px) / entry_px * 100 - 0.20
                trades.append({
                    'date': d, 'day_ret': day_ret, 'net_ret': net_ret
                })

        tdf = pd.DataFrame(trades)
        if not tdf.empty:
            wr = (tdf['net_ret'] > 0).mean() * 100
            print(f"[{entry_t} 진입] 총 거래: {len(tdf)}건 | 승률: {wr:.1f}% | 평균수익: {tdf['net_ret'].mean():+.2f}% | 누적수익: {((1+tdf['net_ret']/100).prod()-1)*100:+.2f}%")
            
            # Filter by day_ret < 0 (day already red)
            red_days = tdf[tdf['day_ret'] < 0]
            if not red_days.empty:
                wr_red = (red_days['net_ret'] > 0).mean() * 100
                print(f"   ↳ 당일 음봉(Day Red) 필터 적용 시 (N={len(red_days)}): 승률 {wr_red:.1f}% | 평균수익 {red_days['net_ret'].mean():+.2f}% | 누적 {((1+red_days['net_ret']/100).prod()-1)*100:+.2f}%")

if __name__ == '__main__':
    explore_variations()
