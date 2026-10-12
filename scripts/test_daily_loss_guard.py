import sqlite3
import config
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.heterogeneous_models import CrossAssetDislocationModel
from core.ml_engine import MLFeatureEngine
from config import DATA_DIR

def test_daily_loss_guard():
    conn = sqlite3.connect(DATA_DIR / 'market_data.db')
    long_15m = pd.read_sql_query("SELECT datetime, open, high, low, close, volume FROM market_candles WHERE symbol=config.BASE_ASSET_LONG AND timeframe='15m' ORDER BY datetime ASC", conn)
    short_15m = pd.read_sql_query("SELECT datetime, open, high, low, close, volume FROM market_candles WHERE symbol=config.BASE_ASSET_SHORT AND timeframe='15m' ORDER BY datetime ASC", conn)
    macro_60m = pd.read_sql_query("SELECT datetime, open, high, low, close, volume FROM market_candles WHERE symbol=config.MACRO_TREND_SYMBOL AND timeframe='60m' ORDER BY datetime ASC", conn)
    long_60m = pd.read_sql_query("SELECT datetime, open, high, low, close, volume FROM market_candles WHERE symbol=config.BASE_ASSET_LONG AND timeframe='60m' ORDER BY datetime ASC", conn)
    vix_15m  = pd.read_sql_query("SELECT datetime, open, high, low, close, volume FROM market_candles WHERE symbol='^VIX' AND timeframe='15m' ORDER BY datetime ASC", conn)
    nvda_15m = pd.read_sql_query("SELECT datetime, open, high, low, close, volume FROM market_candles WHERE symbol='NVDA' AND timeframe='15m' ORDER BY datetime ASC", conn)
    qqq_15m  = pd.read_sql_query("SELECT datetime, open, high, low, close, volume FROM market_candles WHERE symbol='QQQ' AND timeframe='15m' ORDER BY datetime ASC", conn)
    conn.close()

    for df in [long_15m, short_15m, macro_60m, long_60m, vix_15m, nvda_15m, qqq_15m]:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        df.set_index('datetime', inplace=True)

    macro_60m['ema20'] = macro_60m['Close'].ewm(span=20, adjust=False).mean()
    long_60m['ema20'] = long_60m['Close'].ewm(span=20, adjust=False).mean()

    ml_engine = MLFeatureEngine(confidence_threshold=0.60)
    long_15m_feat = ml_engine.extract_features(long_15m)
    short_15m_feat = ml_engine.extract_features(short_15m)
    long_15m_feat['date_str'] = long_15m_feat.index.strftime('%Y-%m-%d')
    unique_dates = sorted(long_15m_feat['date_str'].unique())

    cross_mod = CrossAssetDislocationModel(dislocation_z_threshold=1.6)

    capital = 100000.0
    trades = []
    equity_curve = [capital]

    for d_str in unique_dates:
        day_long = long_15m_feat[long_15m_feat['date_str'] == d_str]
        if len(day_long) < 5:
            continue
        active_pos = None
        day_losses = 0

        for b_idx in range(len(day_long)):
            cur_time = day_long.index[b_idx]
            row_l = day_long.iloc[b_idx]
            time_str = cur_time.strftime('%H:%M')

            if active_pos is not None:
                sym = active_pos['symbol']
                buy_px = active_pos['buy_price']
                bars_held = b_idx - active_pos['entry_bar_idx']
                if sym == config.BASE_ASSET_LONG:
                    cur_h, cur_l, cur_c = row_l['High'], row_l['Low'], row_l['Close']
                else:
                    if cur_time in short_15m_feat.index:
                        r_s = short_15m_feat.loc[cur_time]
                        cur_h, cur_l, cur_c = r_s['High'], r_s['Low'], r_s['Close']
                    else:
                        cur_h = cur_l = cur_c = buy_px

                max_ret = (cur_h - buy_px) / buy_px
                min_ret = (cur_l - buy_px) / buy_px
                exit_triggered = False
                exit_price = 0.0

                if max_ret >= 0.035:
                    exit_triggered = True
                    exit_price = round(buy_px * 1.035 - 0.03, 2)
                elif min_ret <= -0.020:
                    exit_triggered = True
                    exit_price = round(buy_px * 0.98 - 0.03, 2)
                elif bars_held >= 6 or b_idx >= len(day_long) - 1 or time_str >= '15:45':
                    exit_triggered = True
                    exit_price = round(cur_c - 0.03, 2)

                if exit_triggered:
                    pnl = (active_pos['shares'] * (exit_price - buy_px)) - (active_pos['invested'] * 0.0001)
                    capital += pnl
                    equity_curve.append(capital)
                    trades.append(pnl)
                    if pnl <= 0:
                        day_losses += 1
                    active_pos = None

            # 당일 1회 손실 발생 시 당일 추가 매매 즉시 종료 (데일리 락아웃)
            if day_losses >= 1:
                continue

            if active_pos is None and time_str <= '14:30' and b_idx >= 1:
                past_macro_60 = macro_60m[macro_60m.index <= cur_time]
                past_long_60 = long_60m[long_60m.index <= cur_time]
                is_60m_bull = is_60m_bear = False
                if len(past_macro_60) >= 20 and len(past_long_60) >= 20:
                    last_macro = past_macro_60.iloc[-1]
                    last_long = past_long_60.iloc[-1]
                    is_60m_bull = (last_macro['Close'] >= last_macro['ema20'] * 0.998) and (last_long['Close'] >= last_long['ema20'] * 0.998)
                    is_60m_bear = (last_macro['Close'] <= last_macro['ema20'] * 1.002)

                vwap_diff_l = float(row_l.get('VWAP_Diff', 0.0))
                rsi_14_l = float(row_l.get('RSI_14', 50.0))
                bb_lower_l = float(row_l.get('BB_Lower', 0.0))
                cur_close_l = float(row_l['Close'])
                dip_ok_long = (vwap_diff_l <= 1.5) and (rsi_14_l <= 62.0) and (cur_close_l >= bb_lower_l * 1.001 if bb_lower_l > 0 else True)

                dip_ok_short = False
                if cur_time in short_15m_feat.index:
                    row_s = short_15m_feat.loc[cur_time]
                    vwap_diff_s = float(row_s.get('VWAP_Diff', 0.0))
                    rsi_14_s = float(row_s.get('RSI_14', 50.0))
                    bb_lower_s = float(row_s.get('BB_Lower', 0.0))
                    cur_close_s = float(row_s['Close'])
                    dip_ok_short = (vwap_diff_s <= 1.5) and (rsi_14_s <= 62.0) and (cur_close_s >= bb_lower_s * 1.001 if bb_lower_s > 0 else True)

                past_long_15m = long_15m_feat[long_15m_feat.index <= cur_time]
                if len(past_long_15m) >= 30:
                    sub_15m = past_long_15m.tail(60)
                    ret_5 = float(sub_15m['Close'].iloc[-1] / sub_15m['Close'].iloc[-5] - 1.0)
                    past_nvda = nvda_15m[nvda_15m.index <= cur_time]
                    past_qqq = qqq_15m[qqq_15m.index <= cur_time]
                    past_vix = vix_15m[vix_15m.index <= cur_time]
                    if len(past_nvda) >= 5 and len(past_qqq) >= 5 and len(past_vix) >= 5:
                        nvda_r = float(past_nvda['Close'].iloc[-1] / past_nvda['Close'].iloc[-5] - 1.0) * 100
                        qqq_r = float(past_qqq['Close'].iloc[-1] / past_qqq['Close'].iloc[-5] - 1.0) * 100
                        vix_r = float(past_vix['Close'].iloc[-1] / past_vix['Close'].iloc[-5] - 1.0) * 100
                        long_r = ret_5 * 100
                        sig_code, exp_conf, _ = cross_mod.predict_signal(
                            long_ret=long_r, nvda_ret=nvda_r, qqq_ret=qqq_r, macro_ret=long_r, vix_ret=vix_r, tnx_ret=0.0
                        )
                        direction = "LONG" if sig_code > 0 else ("SHORT" if sig_code < 0 else 'NONE')
                        pass_3screen = False
                        if direction == "LONG" and is_60m_bull and dip_ok_long:
                            pass_3screen = True
                            winner_sym = config.BASE_ASSET_LONG
                            base_px = cur_close_l
                        elif direction == "SHORT" and is_60m_bear and dip_ok_short:
                            pass_3screen = True
                            winner_sym = config.BASE_ASSET_SHORT
                            base_px = float(short_15m_feat.loc[cur_time]['Close']) if cur_time in short_15m_feat.index else 40.0

                        if pass_3screen:
                            entry_px = round(base_px + 0.03, 2)
                            shares = int(capital / entry_px)
                            invested = shares * entry_px
                            if shares > 0 and invested > 0:
                                active_pos = {'symbol': winner_sym, 'entry_bar_idx': b_idx, 'buy_price': entry_px, 'shares': shares, 'invested': invested}

    eq_s = pd.Series(equity_curve)
    mdd = abs(((eq_s - eq_s.cummax()) / eq_s.cummax()).min()) * 100
    w_cnt = sum(1 for p in trades if p > 0)
    print(f"=== 🛡️ 당일 1회 손절 시 당일 매매 올스톱 가드 적용 결과 ===")
    print(f"• 총 거래횟수: {len(trades)}회")
    print(f"• 승률: {w_cnt / len(trades) * 100:.2f}% ({w_cnt}승 / {len(trades)-w_cnt}패)")
    print(f"• 총 수익률: {((capital - 100000) / 100000) * 100:+.2f}%")
    print(f"• 순익: ${capital - 100000:+,.2f} USD")
    print(f"• MDD: {mdd:.2f}%")

if __name__ == '__main__':
    test_daily_loss_guard()
