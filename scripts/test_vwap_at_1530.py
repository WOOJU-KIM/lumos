import sqlite3
import sys
import config
import pandas as pd
import numpy as np

if sys.platform.startswith('win'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass

def test_vwap_filter_1530():
    conn = sqlite3.connect('data/market_data.db')
    df_5m = pd.read_sql_query("SELECT symbol, datetime, open, high, low, close, volume FROM market_candles WHERE timeframe='5m' AND symbol IN (config.BASE_ASSET_LONG, config.BASE_ASSET_SHORT)", conn)
    df_5m['datetime'] = pd.to_datetime(df_5m['datetime'])
    df_5m['date'] = df_5m['datetime'].dt.strftime('%Y-%m-%d')
    df_5m['time'] = df_5m['datetime'].dt.strftime('%H:%M')

    days = sorted(df_5m['date'].unique())
    
    trades = []
    TP = 0.015
    SL = -0.010
    SLIP = 0.03
    FEE = 0.0020

    for d in days:
        sub_l = df_5m[(df_5m['date'] == d) & (df_5m['symbol'] == config.BASE_ASSET_LONG)].sort_values('time').copy()
        sub_s = df_5m[(df_5m['date'] == d) & (df_5m['symbol'] == config.BASE_ASSET_SHORT)].sort_values('time').copy()
        if len(sub_l) < 50 or len(sub_s) < 50: continue

        # Compute intraday VWAP for TQQQ
        sub_l['cum_vol'] = sub_l['volume'].cumsum()
        sub_l['cum_pv'] = (sub_l['close'] * sub_l['volume']).cumsum()
        sub_l['vwap'] = sub_l['cum_pv'] / (sub_l['cum_vol'] + 1e-6)

        avg_vol = sub_l[sub_l['time'] < '15:30']['volume'].mean()
        b_1530 = sub_l[sub_l['time'] == '15:30']
        if b_1530.empty: continue
        r1530 = b_1530.iloc[0]

        vol_ratio = r1530['volume'] / (avg_vol + 1e-6)
        bar_ret = (r1530['close'] - r1530['open']) / r1530['open'] * 100
        vwap_diff = (r1530['close'] - r1530['vwap']) / r1530['vwap'] * 100

        post_l = sub_l[(sub_l['time'] > '15:30') & (sub_l['time'] <= '15:50')]
        post_s = sub_s[(sub_s['time'] > '15:30') & (sub_s['time'] <= '15:50')]
        if post_l.empty or post_s.empty: continue

        # Signal Candidate
        if vol_ratio >= 2.0 and bar_ret >= 0.5:
            # Bullish spike in TQQQ
            e_px = post_l.iloc[0]['open'] + SLIP
            tp_px = e_px * (1 + TP)
            sl_px = e_px * (1 + SL)
            exit_px = post_l.iloc[-1]['close'] - SLIP
            reason = "15:50 EOD"
            for k in range(len(post_l)):
                bar = post_l.iloc[k]
                if bar['high'] >= tp_px:
                    exit_px = tp_px - SLIP; reason = "TP +1.5%"; break
                elif bar['low'] <= sl_px:
                    exit_px = sl_px - SLIP; reason = "SL -1.0%"; break
            net_ret = (exit_px - e_px) / e_px * 100 - (FEE * 100)
            trades.append({
                'date': d, 'sym': config.BASE_ASSET_LONG, 'vol_ratio': round(vol_ratio, 1), 'bar_ret': round(bar_ret, 2),
                'vwap_diff': round(vwap_diff, 2), 'vwap_aligned': vwap_diff > 0,
                'net_ret': round(net_ret, 2), 'reason': reason
            })
        elif vol_ratio >= 2.0 and bar_ret <= -0.5:
            # Bearish spike in TQQQ -> Buy SQQQ
            e_px = post_s.iloc[0]['open'] + SLIP
            tp_px = e_px * (1 + TP)
            sl_px = e_px * (1 + SL)
            exit_px = post_s.iloc[-1]['close'] - SLIP
            reason = "15:50 EOD"
            for k in range(len(post_s)):
                bar = post_s.iloc[k]
                if bar['high'] >= tp_px:
                    exit_px = tp_px - SLIP; reason = "TP +1.5%"; break
                elif bar['low'] <= sl_px:
                    exit_px = sl_px - SLIP; reason = "SL -1.0%"; break
            net_ret = (exit_px - e_px) / e_px * 100 - (FEE * 100)
            trades.append({
                'date': d, 'sym': config.BASE_ASSET_SHORT, 'vol_ratio': round(vol_ratio, 1), 'bar_ret': round(bar_ret, 2),
                'vwap_diff': round(vwap_diff, 2), 'vwap_aligned': vwap_diff < 0,
                'net_ret': round(net_ret, 2), 'reason': reason
            })

    tdf = pd.DataFrame(trades)
    print(f"총 15:30 MOC 수급 폭발 거래: {len(tdf)}건")
    print("\n--- [전체 거래 목록 및 VWAP 정렬 여부] ---")
    print(tdf.to_string(index=False))

    print("\n--- [비교: VWAP 필터 적용 vs 미적용] ---")
    # All
    wr_all = (tdf['net_ret'] > 0).mean() * 100
    print(f"[1. VWAP 미적용 전체]: 거래 {len(tdf)}회 | 승률 {wr_all:.1f}% (승 {(tdf['net_ret']>0).sum()}/패 {(tdf['net_ret']<=0).sum()}) | 평균수익 {tdf['net_ret'].mean():+.2f}% | 누적 {((1+tdf['net_ret']/100).prod()-1)*100:+.2f}%")

    # VWAP aligned (Price > VWAP for TQQQ, Price < VWAP for SQQQ)
    aligned = tdf[tdf['vwap_aligned'] == True]
    wr_alg = (aligned['net_ret'] > 0).mean() * 100
    print(f"[2. VWAP 정렬(순방향) 필터 적용]: 거래 {len(aligned)}회 | 승률 {wr_alg:.1f}% (승 {(aligned['net_ret']>0).sum()}/패 {(aligned['net_ret']<=0).sum()}) | 평균수익 {aligned['net_ret'].mean():+.2f}% | 누적 {((1+aligned['net_ret']/100).prod()-1)*100:+.2f}%")

    # VWAP unaligned (Against VWAP)
    unaligned = tdf[tdf['vwap_aligned'] == False]
    wr_un = (unaligned['net_ret'] > 0).mean() * 100
    print(f"[3. VWAP 역방향(불일치)]: 거래 {len(unaligned)}회 | 승률 {wr_un:.1f}% (승 {(unaligned['net_ret']>0).sum()}/패 {(unaligned['net_ret']<=0).sum()}) | 평균수익 {unaligned['net_ret'].mean():+.2f}% | 누적 {((1+unaligned['net_ret']/100).prod()-1)*100:+.2f}%")

if __name__ == '__main__':
    test_vwap_filter_1530()
