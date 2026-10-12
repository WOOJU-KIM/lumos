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

def deep_dive_panic_dump():
    conn = sqlite3.connect('data/market_data.db')
    
    # Load TQQQ and SQQQ 5m candles
    df_long = pd.read_sql_query("""
        SELECT datetime, open, high, low, close, volume 
        FROM market_candles 
        WHERE symbol=config.BASE_ASSET_LONG AND timeframe='5m' 
        ORDER BY datetime ASC
    """, conn)
    df_long['datetime'] = pd.to_datetime(df_long['datetime'])
    df_long['date'] = df_long['datetime'].dt.strftime('%Y-%m-%d')
    df_long['time'] = df_long['datetime'].dt.strftime('%H:%M')

    df_short = pd.read_sql_query("""
        SELECT datetime, open, high, low, close, volume 
        FROM market_candles 
        WHERE symbol=config.BASE_ASSET_SHORT AND timeframe='5m' 
        ORDER BY datetime ASC
    """, conn)
    df_short['datetime'] = pd.to_datetime(df_short['datetime'])
    df_short['date'] = df_short['datetime'].dt.strftime('%Y-%m-%d')
    df_short['time'] = df_short['datetime'].dt.strftime('%H:%M')

    days = sorted(df_long['date'].unique())
    trades = []

    for d in days:
        sub_l = df_long[df_long['date'] == d].sort_values('time').copy()
        sub_s = df_short[df_short['date'] == d].sort_values('time').copy()
        if len(sub_l) < 50 or len(sub_s) < 50:
            continue

        # 1. Look at 14:00~15:00 box in TQQQ
        box_14_15 = sub_l[(sub_l['time'] >= '14:00') & (sub_l['time'] < '15:00')]
        if len(box_14_15) < 10:
            continue
        box_l_low = box_14_15['low'].min()
        box_l_high = box_14_15['high'].max()

        # 2. Check 15:00 candle in TQQQ
        bar_1500_l = sub_l[sub_l['time'] == '15:00']
        bar_1500_s = sub_s[sub_s['time'] == '15:00']
        if bar_1500_l.empty or bar_1500_s.empty:
            continue

        c_1500_l = bar_1500_l.iloc[0]['close']
        c_1500_s = bar_1500_s.iloc[0]['close']

        # Condition: 15:00 close in TQQQ broke down below 14:00~15:00 box low
        if c_1500_l < box_l_low:
            # We buy SQQQ at 15:00 close (with 0.03 slippage)
            entry_px = c_1500_s + 0.03
            
            # Post 15:00 to 15:50 in SQQQ
            post_s = sub_s[(sub_s['time'] > '15:00') & (sub_s['time'] <= '15:50')]
            if post_s.empty:
                continue

            # Evaluate intra-bar TP/SL or pure 15:50 exit
            # Option A: Pure 15:50 exit
            exit_1550_s = post_s.iloc[-1]['close'] - 0.03
            ret_raw = (exit_1550_s - entry_px) / entry_px * 100.0 - 0.20 # net fee
            
            # Max run-up / run-down in SQQQ during 15:00~15:50
            max_short_px = post_s['high'].max()
            min_short_px = post_s['low'].min()
            max_run_up = (max_short_px - entry_px) / entry_px * 100.0
            max_drawdown = (min_short_px - entry_px) / entry_px * 100.0

            # Option B: With TP +2.0% / SL -1.5%
            tp_px = entry_px * 1.020
            sl_px = entry_px * 0.985
            exit_b_px = exit_1550_s
            exit_reason = "15:50 EOD"
            for k in range(len(post_s)):
                bar = post_s.iloc[k]
                h, l, c = bar['high'], bar['low'], bar['close']
                if h >= tp_px:
                    exit_b_px = tp_px - 0.03
                    exit_reason = "TP +2.0%"
                    break
                elif l <= sl_px:
                    exit_b_px = sl_px - 0.03
                    exit_reason = "SL -1.5%"
                    break
            ret_b = (exit_b_px - entry_px) / entry_px * 100.0 - 0.20

            trades.append({
                'date': d,
                'long_box_low': round(box_l_low, 2),
                'long_1500': round(c_1500_l, 2),
                'short_entry': round(entry_px, 2),
                'short_exit_1550': round(exit_1550_s, 2),
                'ret_pure_1550': round(ret_raw, 2),
                'ret_tp_sl': round(ret_b, 2),
                'exit_reason': exit_reason,
                'max_up': round(max_run_up, 2),
                'max_dn': round(max_drawdown, 2)
            })

    res_df = pd.DataFrame(trades)
    print(f"=== [패닉 덤프 SQQQ 스나이퍼 상세 백테스트 내역 (총 {len(res_df)}건)] ===")
    print(res_df.to_string(index=False))

    print("\n--- [성과 요약 1: 순수 15:50 장 마감 청산] ---")
    wins_pure = (res_df['ret_pure_1550'] > 0).sum()
    total = len(res_df)
    wr_pure = wins_pure / total * 100.0
    avg_ret_pure = res_df['ret_pure_1550'].mean()
    cum_ret_pure = ((1 + res_df['ret_pure_1550']/100.0).prod() - 1) * 100.0
    print(f"총 거래: {total}회 | 승: {wins_pure}회 | 패: {total - wins_pure}회")
    print(f"순수 15:50 청산 승률: {wr_pure:.1f}%")
    print(f"건당 평균 순수익률: {avg_ret_pure:+.2f}%")
    print(f"누적 복리 수익률: {cum_ret_pure:+.2f}%")

    print("\n--- [성과 요약 2: TP +2.0% / SL -1.5% 적용 시] ---")
    wins_b = (res_df['ret_tp_sl'] > 0).sum()
    wr_b = wins_b / total * 100.0
    avg_ret_b = res_df['ret_tp_sl'].mean()
    cum_ret_b = ((1 + res_df['ret_tp_sl']/100.0).prod() - 1) * 100.0
    print(f"총 거래: {total}회 | 승: {wins_b}회 | 패: {total - wins_b}회")
    print(f"TP/SL 적용 승률: {wr_b:.1f}%")
    print(f"건당 평균 순수익률: {avg_ret_b:+.2f}%")
    print(f"누적 복리 수익률: {cum_ret_b:+.2f}%")
    print("청산 사유 분포:\n", res_df['exit_reason'].value_counts().to_string())

if __name__ == '__main__':
    deep_dive_panic_dump()
