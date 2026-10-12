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

def test_cross_asset_at_close():
    conn = sqlite3.connect('data/market_data.db')
    
    # Load 5m TQQQ, QQQ, NVDA, ^VIX
    query = """
        SELECT symbol, datetime, open, high, low, close, volume 
        FROM market_candles 
        WHERE timeframe='5m' AND symbol IN (config.BASE_ASSET_LONG, 'QQQ', 'NVDA', '^VIX')
        ORDER BY datetime ASC
    """
    df = pd.read_sql_query(query, conn)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')
    df['time'] = df['datetime'].dt.strftime('%H:%M')

    days = sorted(df['date'].unique())
    results = []

    for d in days:
        day_data = df[df['date'] == d]
        
        long_5 = day_data[day_data['symbol'] == config.BASE_ASSET_LONG]
        qqq_5 = day_data[day_data['symbol'] == 'QQQ']
        nvda_5 = day_data[day_data['symbol'] == 'NVDA']
        vix_5 = day_data[day_data['symbol'] == '^VIX']

        if long_5.empty or qqq_5.empty or nvda_5.empty:
            continue

        p_1500_long = long_5[long_5['time'] == '15:00']
        p_1550_long = long_5[long_5['time'] == '15:50']
        p_1430_long = long_5[long_5['time'] == '14:30']

        p_1500_qqq = qqq_5[qqq_5['time'] == '15:00']
        p_1430_qqq = qqq_5[qqq_5['time'] == '14:30']

        p_1500_nvda = nvda_5[nvda_5['time'] == '15:00']
        p_1430_nvda = nvda_5[nvda_5['time'] == '14:30']

        if p_1500_long.empty or p_1550_long.empty or p_1430_long.empty or p_1500_qqq.empty or p_1430_qqq.empty or p_1500_nvda.empty or p_1430_nvda.empty:
            continue

        s1500 = p_1500_long.iloc[0]['close']
        s1550 = p_1550_long.iloc[0]['close']
        s1430 = p_1430_long.iloc[0]['close']

        q1500 = p_1500_qqq.iloc[0]['close']
        q1430 = p_1430_qqq.iloc[0]['close']

        n1500 = p_1500_nvda.iloc[0]['close']
        n1430 = p_1430_nvda.iloc[0]['close']

        # Return between 14:30 and 15:00
        long_pwr_in = (s1500 - s1430) / s1430 * 100
        qqq_pwr_in = (q1500 - q1430) / q1430 * 100
        nvda_pwr_in = (n1500 - n1430) / n1430 * 100

        # Outcome: 15:00 to 15:50 TQQQ return
        long_out = (s1550 - s1500) / s1500 * 100

        results.append({
            'date': d,
            'long_pwr_in': long_pwr_in,
            'qqq_pwr_in': qqq_pwr_in,
            'nvda_pwr_in': nvda_pwr_in,
            'long_out': long_out
        })

    res = pd.DataFrame(results)
    print(f"유효 분석일: {len(res)}일")

    # Triple confirmation at 15:00: Both QQQ, NVDA, TQQQ fell between 14:30 and 15:00
    all_down = res[(res['qqq_pwr_in'] < -0.2) & (res['nvda_pwr_in'] < -0.3) & (res['long_pwr_in'] < -0.5)]
    print(f"\n[크로스에셋 하락 일치: 14:30~15:00 QQQ < -0.2% & NVDA < -0.3% & TQQQ < -0.5% (N={len(all_down)}일)]")
    # SQQQ profit
    short_win = (all_down['long_out'] < 0).mean() * 100
    print(f"  - 15:00~15:50 SQQQ(추가 하락) 승률: {short_win:.1f}%")
    print(f"  - SQQQ 평균 수익률: {-all_down['long_out'].mean():+.2f}%")

    # Triple confirmation at 15:00: Both QQQ, NVDA, TQQQ rose between 14:30 and 15:00
    all_up = res[(res['qqq_pwr_in'] > 0.2) & (res['nvda_pwr_in'] > 0.3) & (res['long_pwr_in'] > 0.5)]
    print(f"\n[크로스에셋 상승 일치: 14:30~15:00 QQQ > 0.2% & NVDA > 0.3% & TQQQ > 0.5% (N={len(all_up)}일)]")
    long_win = (all_up['long_out'] > 0).mean() * 100
    print(f"  - 15:00~15:50 TQQQ(추가 상승) 승률: {long_win:.1f}%")
    print(f"  - TQQQ 평균 수익률: {all_up['long_out'].mean():+.2f}%")

if __name__ == '__main__':
    test_cross_asset_at_close()
