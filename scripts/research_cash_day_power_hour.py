import sqlite3
import sys
import config
import pandas as pd
import numpy as np

def test_fading():
    conn = sqlite3.connect('data/market_data.db')
    df_5m = pd.read_sql_query("SELECT symbol, datetime, open, high, low, close, volume FROM market_candles WHERE timeframe='5m' AND symbol IN (config.BASE_ASSET_LONG, config.BASE_ASSET_SHORT)", conn)
    df_5m['datetime'] = pd.to_datetime(df_5m['datetime'])
    df_5m['date'] = df_5m['datetime'].dt.strftime('%Y-%m-%d')
    df_5m['time'] = df_5m['datetime'].dt.strftime('%H:%M')
    unique_dates = sorted(df_5m['date'].unique())

    TP = 0.015
    SL = -0.010
    SLIP = 0.03
    FEE = 0.0020

    print("=== [14:30 역추세 페이딩(Fade the Move) 전략: 오른 놈 숏 치고 내린 놈 롱 치기] ===")
    for min_trend in [1.5, 2.0, 2.5, 3.0]:
        results = []
        for d in unique_dates:
            sub_l = df_5m[(df_5m['date'] == d) & (df_5m['symbol'] == config.BASE_ASSET_LONG)].sort_values('time')
            sub_s = df_5m[(df_5m['date'] == d) & (df_5m['symbol'] == config.BASE_ASSET_SHORT)].sort_values('time')
            if len(sub_l) < 50 or len(sub_s) < 50: continue

            b_open = sub_l[sub_l['time'] == '09:30']
            b_1430_l = sub_l[sub_l['time'] == '14:30']
            b_1430_s = sub_s[sub_s['time'] == '14:30']
            if b_open.empty or b_1430_l.empty or b_1430_s.empty: continue

            d_open = b_open.iloc[0]['open']
            c_1430_l = b_1430_l.iloc[0]['close']
            c_1430_s = b_1430_s.iloc[0]['close']

            day_ret = (c_1430_l - d_open) / d_open * 100

            post_l = sub_l[(sub_l['time'] > '14:30') & (sub_l['time'] <= '15:50')]
            post_s = sub_s[(sub_s['time'] > '14:30') & (sub_s['time'] <= '15:50')]
            if post_l.empty or post_s.empty: continue

            sym = None
            target_post = None
            entry_base = None

            # FADING: If up > min_trend -> Buy SQQQ (Short TQQQ)
            if day_ret >= min_trend:
                sym = config.BASE_ASSET_SHORT; target_post = post_s; entry_base = c_1430_s
            # If down < -min_trend -> Buy TQQQ (Long TQQQ)
            elif day_ret <= -min_trend:
                sym = config.BASE_ASSET_LONG; target_post = post_l; entry_base = c_1430_l

            if sym is None: continue

            entry_px = entry_base + SLIP
            tp_px = entry_px * (1 + TP)
            sl_px = entry_px * (1 + SL)
            exit_px = target_post.iloc[-1]['close'] - SLIP
            reason = "15:50 EOD"

            for k in range(len(target_post)):
                bar = target_post.iloc[k]
                h, l, c = bar['high'], bar['low'], bar['close']
                if h >= tp_px:
                    exit_px = tp_px - SLIP
                    reason = "TP +1.5%"
                    break
                elif l <= sl_px:
                    exit_px = sl_px - SLIP
                    reason = "SL -1.0%"
                    break

            net_ret = (exit_px - entry_px) / entry_px * 100 - (FEE * 100)
            results.append({
                'date': d, 'sym': sym, 'day_ret': round(day_ret, 2),
                'net_ret': round(net_ret, 2), 'reason': reason
            })

        res_df = pd.DataFrame(results)
        if not res_df.empty:
            wr = (res_df['net_ret'] > 0).mean() * 100
            cum = ((1 + res_df['net_ret']/100).prod() - 1) * 100
            print(f"기준 당일추세 >={min_trend}% 페이딩: 거래 {len(res_df)}회 | 승률 {wr:.1f}% (승 {(res_df['net_ret']>0).sum()}/패 {(res_df['net_ret']<=0).sum()}) | 건당수익 {res_df['net_ret'].mean():+.2f}% | 누적 {cum:+.2f}%")

if __name__ == '__main__':
    test_fading()
