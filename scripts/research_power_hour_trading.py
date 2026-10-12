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

def research_power_hour():
    conn = sqlite3.connect('data/market_data.db')
    
    # Load 5m and 15m TQQQ, SQQQ, QQQ, NVDA
    query_5m = """
        SELECT symbol, datetime, open, high, low, close, volume 
        FROM market_candles 
        WHERE timeframe='5m' AND symbol IN (config.BASE_ASSET_LONG, config.BASE_ASSET_SHORT, 'QQQ', 'NVDA')
        ORDER BY datetime ASC
    """
    df_5m = pd.read_sql_query(query_5m, conn)
    df_5m['datetime'] = pd.to_datetime(df_5m['datetime'])
    df_5m['date'] = df_5m['datetime'].dt.strftime('%Y-%m-%d')
    df_5m['time'] = df_5m['datetime'].dt.strftime('%H:%M')

    days = sorted(df_5m['date'].unique())
    print(f"총 유효 분석 거래일: {len(days)}일")

    # =========================================================================
    # Strategy 1: [15:30 MOC 수급 폭발 추종 20분 스나이퍼 (15:30 진입 ➔ 15:50 청산)]
    # 조건: 15:30 5분봉의 거래량이 당일 평균의 2.5배 이상 폭증하고 강한 양봉/음봉 형성 시
    # =========================================================================
    strat1_trades = []
    for d in days:
        sub = df_5m[(df_5m['date'] == d) & (df_5m['symbol'] == config.BASE_ASSET_LONG)].sort_values('time')
        sub_s = df_5m[(df_5m['date'] == d) & (df_5m['symbol'] == config.BASE_ASSET_SHORT)].sort_values('time')
        if len(sub) < 50 or len(sub_s) < 50: continue

        avg_vol = sub[sub['time'] < '15:30']['volume'].mean()
        b_1530 = sub[sub['time'] == '15:30']
        if b_1530.empty: continue
        r1530 = b_1530.iloc[0]

        vol_ratio = r1530['volume'] / (avg_vol + 1e-6)
        bar_ret = (r1530['close'] - r1530['open']) / r1530['open'] * 100

        # Post 15:30 to 15:50
        post_l = sub[(sub['time'] > '15:30') & (sub['time'] <= '15:50')]
        post_s = sub_s[(sub_s['time'] > '15:30') & (sub_s['time'] <= '15:50')]
        if post_l.empty or post_s.empty: continue

        # If 15:30 volume >= 2.0x and bar_ret >= +0.5% -> Buy TQQQ
        if vol_ratio >= 2.0 and bar_ret >= 0.5:
            e_px = post_l.iloc[0]['open'] + 0.03
            exit_px = post_l.iloc[-1]['close'] - 0.03
            ret = (exit_px - e_px) / e_px * 100 - 0.20
            strat1_trades.append({'date': d, 'sym': config.BASE_ASSET_LONG, 'ret': ret, 'vol_ratio': vol_ratio, 'bar_ret': bar_ret})
        # If 15:30 volume >= 2.0x and bar_ret <= -0.5% -> Buy SQQQ
        elif vol_ratio >= 2.0 and bar_ret <= -0.5:
            e_px = post_s.iloc[0]['open'] + 0.03
            exit_px = post_s.iloc[-1]['close'] - 0.03
            ret = (exit_px - e_px) / e_px * 100 - 0.20
            strat1_trades.append({'date': d, 'sym': config.BASE_ASSET_SHORT, 'ret': ret, 'vol_ratio': vol_ratio, 'bar_ret': bar_ret})

    s1_df = pd.DataFrame(strat1_trades)
    print("\n" + "="*70)
    print("전략 1: [15:30 MOC 수급 폭발 추종 20분 스나이퍼]")
    print("규칙: 15:30 5분봉 거래량 2배 폭증 + 0.5% 이상 장대봉 출현 시 해당 방향 진입 ➔ 15:50 청산")
    print("="*70)
    if not s1_df.empty:
        wr1 = (s1_df['ret'] > 0).mean() * 100
        print(f"총 거래: {len(s1_df)}건 | 승: {(s1_df['ret']>0).sum()}건 | 패: {(s1_df['ret']<=0).sum()}건")
        print(f"승률: {wr1:.1f}% | 건당 평균 수익: {s1_df['ret'].mean():+.2f}% | 누적 수익: {((1+s1_df['ret']/100).prod()-1)*100:+.2f}%")
        print(s1_df[['date', 'sym', 'ret', 'vol_ratio', 'bar_ret']].to_string(index=False))

    # =========================================================================
    # Strategy 2: [15:00 파워아워 30분 모멘텀 추종 (15:00 진입 ➔ 15:30 또는 15:50 청산)]
    # 조건: 14:00~15:00 박스권 상향/하향 강력 돌파 + QQQ/NVDA 방향 일치
    # =========================================================================
    strat2_trades = []
    for d in days:
        sub_l = df_5m[(df_5m['date'] == d) & (df_5m['symbol'] == config.BASE_ASSET_LONG)].sort_values('time')
        sub_s = df_5m[(df_5m['date'] == d) & (df_5m['symbol'] == config.BASE_ASSET_SHORT)].sort_values('time')
        sub_q = df_5m[(df_5m['date'] == d) & (df_5m['symbol'] == 'QQQ')].sort_values('time')
        if len(sub_l) < 50 or len(sub_s) < 50 or len(sub_q) < 50: continue

        # 14:00~15:00 box in TQQQ
        box_l = sub_l[(sub_l['time'] >= '14:00') & (sub_l['time'] < '15:00')]
        box_h = box_l['high'].max()
        box_low = box_l['low'].min()

        b1500_l = sub_l[sub_l['time'] == '15:00']
        b1500_s = sub_s[sub_s['time'] == '15:00']
        b1500_q = sub_q[sub_q['time'] == '15:00']
        b1400_q = sub_q[sub_q['time'] == '14:00']
        if b1500_l.empty or b1500_s.empty or b1500_q.empty or b1400_q.empty: continue

        c_l = b1500_l.iloc[0]['close']
        c_s = b1500_s.iloc[0]['close']
        q_ret_14_15 = (b1500_q.iloc[0]['close'] - b1400_q.iloc[0]['close']) / b1400_q.iloc[0]['close'] * 100

        post_l = sub_l[(sub_l['time'] > '15:00') & (sub_l['time'] <= '15:50')]
        post_s = sub_s[(sub_s['time'] > '15:00') & (sub_s['time'] <= '15:50')]
        if post_l.empty or post_s.empty: continue

        # Up breakout + QQQ positive
        if c_l > box_h and q_ret_14_15 > 0.1:
            e_px = c_l + 0.03
            exit_px = post_l.iloc[-1]['close'] - 0.03
            ret = (exit_px - e_px) / e_px * 100 - 0.20
            strat2_trades.append({'date': d, 'sym': config.BASE_ASSET_LONG, 'ret': ret, 'type': 'UP_BREAK'})
        # Down breakdown (any QQQ)
        elif c_l < box_low:
            e_px = c_s + 0.03
            exit_px = post_s.iloc[-1]['close'] - 0.03
            ret = (exit_px - e_px) / e_px * 100 - 0.20
            strat2_trades.append({'date': d, 'sym': config.BASE_ASSET_SHORT, 'ret': ret, 'type': 'DN_BREAK'})

    s2_df = pd.DataFrame(strat2_trades)
    print("\n" + "="*70)
    print("전략 2: [15:00 파워아워 박스권 돌파 추종 스나이퍼]")
    print("="*70)
    if not s2_df.empty:
        wr2 = (s2_df['ret'] > 0).mean() * 100
        print(f"총 거래: {len(s2_df)}건 | 승: {(s2_df['ret']>0).sum()}건 | 패: {(s2_df['ret']<=0).sum()}건")
        print(f"승률: {wr2:.1f}% | 건당 평균 수익: {s2_df['ret'].mean():+.2f}% | 누적 수익: {((1+s2_df['ret']/100).prod()-1)*100:+.2f}%")
        print(s2_df.to_string(index=False))

    # =========================================================================
    # Strategy 3: [15:00 파워아워 당일 고가/저가 인근 '피날레 스퀴즈' (Day Squeeze)]
    # 조건: 15:00 시점에 당일 전체 최고점 대비 -0.5% 이내로 근접해 있거나 최저점 대비 +0.5% 이내일 때
    # =========================================================================
    strat3_trades = []
    for d in days:
        sub_l = df_5m[(df_5m['date'] == d) & (df_5m['symbol'] == config.BASE_ASSET_LONG)].sort_values('time')
        sub_s = df_5m[(df_5m['date'] == d) & (df_5m['symbol'] == config.BASE_ASSET_SHORT)].sort_values('time')
        if len(sub_l) < 50 or len(sub_s) < 50: continue

        prior_l = sub_l[sub_l['time'] <= '15:00']
        day_high = prior_l['high'].max()
        day_low = prior_l['low'].min()

        b1500_l = sub_l[sub_l['time'] == '15:00']
        b1500_s = sub_s[sub_s['time'] == '15:00']
        if b1500_l.empty or b1500_s.empty: continue
        c_l = b1500_l.iloc[0]['close']
        c_s = b1500_s.iloc[0]['close']

        post_l = sub_l[(sub_l['time'] > '15:00') & (sub_l['time'] <= '15:50')]
        post_s = sub_s[(sub_s['time'] > '15:00') & (sub_s['time'] <= '15:50')]
        if post_l.empty or post_s.empty: continue

        # Near day high (within 0.5%): Squeeze TQQQ
        if (day_high - c_l) / c_l * 100 <= 0.5:
            e_px = c_l + 0.03
            exit_px = post_l.iloc[-1]['close'] - 0.03
            ret = (exit_px - e_px) / e_px * 100 - 0.20
            strat3_trades.append({'date': d, 'sym': config.BASE_ASSET_LONG, 'ret': ret, 'type': 'HIGH_SQUEEZE'})
        # Near day low (within 0.5%): Dump SQQQ
        elif (c_l - day_low) / day_low * 100 <= 0.5:
            e_px = c_s + 0.03
            exit_px = post_s.iloc[-1]['close'] - 0.03
            ret = (exit_px - e_px) / e_px * 100 - 0.20
            strat3_trades.append({'date': d, 'sym': config.BASE_ASSET_SHORT, 'ret': ret, 'type': 'LOW_DUMP'})

    s3_df = pd.DataFrame(strat3_trades)
    print("\n" + "="*70)
    print("전략 3: [15:00 당일 고가/저가 피날레 스퀴즈]")
    print("="*70)
    if not s3_df.empty:
        wr3 = (s3_df['ret'] > 0).mean() * 100
        print(f"총 거래: {len(s3_df)}건 | 승: {(s3_df['ret']>0).sum()}건 | 패: {(s3_df['ret']<=0).sum()}건")
        print(f"승률: {wr3:.1f}% | 건당 평균 수익: {s3_df['ret'].mean():+.2f}% | 누적 수익: {((1+s3_df['ret']/100).prod()-1)*100:+.2f}%")

if __name__ == '__main__':
    research_power_hour()
