import os
import sys
import sqlite3
import config
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from core.data_lake import MarketDataLake
from core.ml_engine import MLFeatureEngine
from core.heterogeneous_models import CrossAssetDislocationModel

def analyze_close_riding():
    lake = MarketDataLake()
    print("⏳ 데이터 로드 및 15분봉 챔피언 매매 시뮬레이션 시작...")
    
    long_15m = lake.load_candles(config.BASE_ASSET_LONG, "15m")
    short_15m = lake.load_candles(config.BASE_ASSET_SHORT, "15m")
    macro_60m = lake.load_candles(config.MACRO_TREND_SYMBOL, "60m")
    long_60m = lake.load_candles(config.BASE_ASSET_LONG, "60m")
    long_5m = lake.load_candles(config.BASE_ASSET_LONG, "5m")
    short_5m = lake.load_candles(config.BASE_ASSET_SHORT, "5m")
    macro_15m = lake.load_candles(config.MACRO_TREND_SYMBOL, "15m")
    nvda_15m = lake.load_candles("NVDA", "15m")
    qqq_15m = lake.load_candles("QQQ", "15m")
    vix_15m = lake.load_candles("^VIX", "15m")

    macro_60m['ema20'] = macro_60m['Close'].ewm(span=20, adjust=False).mean()
    long_60m['ema20'] = long_60m['Close'].ewm(span=20, adjust=False).mean()

    for df in [long_15m, short_15m, long_5m, short_5m, macro_60m, long_60m, macro_15m, nvda_15m, qqq_15m, vix_15m]:
        df['datetime_dt'] = pd.to_datetime(df['datetime'])
        df['date_str'] = df['datetime_dt'].dt.strftime('%Y-%m-%d')
        df['time_str'] = df['datetime_dt'].dt.strftime('%H:%M')

    ml_15m = MLFeatureEngine()
    long_15m_feat = ml_15m.extract_features(long_15m)
    long_15m_feat = ml_15m.add_confidence_columns(long_15m_feat)
    short_15m_feat = ml_15m.extract_features(short_15m)
    short_15m_feat.set_index('datetime', inplace=True, drop=False)

    cross_mod = CrossAssetDislocationModel(dislocation_z_threshold=1.6)
    nvda_map = nvda_15m.set_index('datetime')['Close'].to_dict()
    macro_map = macro_15m.set_index('datetime')['Close'].to_dict()
    qqq_map = qqq_15m.set_index('datetime')['Close'].to_dict()
    vix_map = vix_15m.set_index('datetime')['Close'].to_dict()
    long_close = long_15m['Close'].values
    long_dt = long_15m['datetime'].values

    cross_dirs = []
    cross_confs = []
    for i in range(len(long_15m)):
        if i < 5:
            cross_dirs.append('NONE'); cross_confs.append(0.50); continue
        c_t, p_t = long_dt[i], long_dt[i-5]
        if c_t in nvda_map and p_t in nvda_map and c_t in qqq_map and p_t in qqq_map and c_t in vix_map and p_t in vix_map:
            s_r = float(long_close[i]/long_close[i-5] - 1.0)
            n_r = float(nvda_map[c_t]/nvda_map[p_t] - 1.0)
            sx_r = float(macro_map[c_t]/macro_map[p_t] - 1.0) if c_t in macro_map and p_t in macro_map else n_r
            q_r = float(qqq_map[c_t]/qqq_map[p_t] - 1.0)
            v_r = float(vix_map[c_t]/vix_map[p_t] - 1.0)
            code, conf, _ = cross_mod.predict_signal(long_ret=s_r, nvda_ret=n_r, macro_ret=sx_r, qqq_ret=q_r, vix_ret=v_r, tnx_ret=0.0)
            c_dir = "LONG" if code > 0 else ("SHORT" if code < 0 else 'NONE')
            cross_dirs.append(c_dir); cross_confs.append(conf)
        else:
            cross_dirs.append('NONE'); cross_confs.append(0.50)

    long_15m_feat['cross_dir'] = cross_dirs
    long_15m_feat['cross_conf'] = cross_confs
    long_15m_feat.set_index('datetime', inplace=True, drop=False)

    unique_dates = sorted(long_15m['date_str'].unique())

    TP_15M = 0.030
    SL_15M = -0.020
    SLIPPAGE = 0.03
    FEE_RATE = 0.0020
    TIME_STOP_BARS = 18

    capital = 10000.0
    trades = []

    for d_str in unique_dates:
        day_15 = long_15m_feat[long_15m_feat['date_str'] == d_str]
        day_5_l = long_5m[long_5m['date_str'] == d_str]
        day_5_s = short_5m[short_5m['date_str'] == d_str]
        if len(day_15) < 5: continue

        b_idx = 0
        n_bars = len(day_15)

        while b_idx < n_bars:
            row_15 = day_15.iloc[b_idx]
            cur_time = row_15['datetime']
            time_str = row_15['time_str']

            if b_idx < 1 or time_str > '14:30':
                b_idx += 1; continue

            dir_gbdt = row_15['Direction']
            conf_gbdt = float(row_15['Confidence'])
            dir_cross = row_15['cross_dir']
            conf_cross = float(row_15['cross_conf'])

            # Hybrid MoE V3 rule (GBDT >= 60%, Veto >= 60%)
            is_gbdt = (dir_gbdt in ["LONG", "SHORT"]) and (conf_gbdt >= 0.60)
            is_opposite_veto = (
                (dir_gbdt == "LONG" and dir_cross == "SHORT" and conf_cross >= 0.60) or
                (dir_gbdt == "SHORT" and dir_cross == "LONG" and conf_cross >= 0.60)
            )
            if not (is_gbdt and not is_opposite_veto):
                b_idx += 1; continue

            chosen_sym = config.BASE_ASSET_LONG if dir_gbdt == "LONG" else config.BASE_ASSET_SHORT

            # Screen 1: 60m trend
            past_macro = macro_60m[macro_60m['datetime'] <= cur_time]
            past_long = long_60m[long_60m['datetime'] <= cur_time]
            if len(past_macro) < 20 or len(past_long) < 20:
                b_idx += 1; continue
            macro_c = past_macro['Close'].iloc[-1]
            long_c = past_long['Close'].iloc[-1]
            macro_ema = past_macro['ema20'].iloc[-1]
            long_ema = past_long['ema20'].iloc[-1]
            if chosen_sym == config.BASE_ASSET_LONG and not (macro_c >= macro_ema * 0.998 and long_c >= long_ema * 0.998):
                b_idx += 1; continue
            elif chosen_sym == config.BASE_ASSET_SHORT and not (macro_c <= macro_ema * 1.002):
                b_idx += 1; continue

            # Screen 3: Dip filter
            if chosen_sym == config.BASE_ASSET_LONG:
                vd = float(row_15.get('VWAP_Diff', 0.0))
                r14 = float(row_15.get('RSI_14', 50.0))
                bbl = float(row_15.get('BB_Lower', 0.0))
                if not (vd <= 1.5 and r14 <= 62.0 and (bbl <= 0 or float(row_15['Close']) >= bbl * 1.001)):
                    b_idx += 1; continue
            else:
                if cur_time not in short_15m_feat.index:
                    b_idx += 1; continue
                rs = short_15m_feat.loc[cur_time]
                vd = float(rs.get('VWAP_Diff', 0.0))
                r14 = float(rs.get('RSI_14', 50.0))
                bbl = float(rs.get('BB_Lower', 0.0))
                if not (vd <= 1.5 and r14 <= 62.0 and (bbl <= 0 or float(rs['Close']) >= bbl * 1.001)):
                    b_idx += 1; continue

            base_px = float(row_15['Close']) if chosen_sym == config.BASE_ASSET_LONG else float(short_15m_feat.loc[cur_time]['Close'])
            entry_px = round(base_px + SLIPPAGE, 2)
            shares = int(capital / entry_px)
            invested = shares * entry_px
            if shares <= 0: b_idx += 1; continue

            target_5m = day_5_l if chosen_sym == config.BASE_ASSET_LONG else day_5_s
            post_5m = target_5m[target_5m['datetime'] > cur_time]
            if post_5m.empty: b_idx += 1; continue

            tp_px = round(entry_px * (1 + TP_15M), 2)
            sl_px = round(entry_px * (1 + SL_15M), 2)
            exit_px = entry_px
            exit_reason = "TimeStop"
            exit_time_str = ""
            bars_held = 0

            eval_5m = post_5m.iloc[:TIME_STOP_BARS]
            for k in range(len(eval_5m)):
                c5 = eval_5m.iloc[k]
                t5 = c5['time_str']
                c5_o, c5_h, c5_l, c5_c = float(c5['Open']), float(c5['High']), float(c5['Low']), float(c5['Close'])
                bars_held = k + 1
                hit_tp, hit_sl = (c5_h >= tp_px), (c5_l <= sl_px)

                if hit_tp and hit_sl:
                    if c5_c >= c5_o:
                        exit_px = round(tp_px - SLIPPAGE, 2); exit_reason = "TP +3.0%"
                    else:
                        exit_px = round(sl_px - SLIPPAGE, 2); exit_reason = "SL -2.0%"
                    exit_time_str = t5
                    break
                elif hit_tp:
                    exit_px = round(tp_px - SLIPPAGE, 2); exit_reason = "TP +3.0%"
                    exit_time_str = t5
                    break
                elif hit_sl:
                    exit_px = round(sl_px - SLIPPAGE, 2); exit_reason = "SL -2.0%"
                    exit_time_str = t5
                    break
                elif t5 >= '15:45' or k == len(eval_5m) - 1:
                    exit_px = round(c5_c - SLIPPAGE, 2)
                    exit_reason = "15:50 EOD" if t5 >= '15:45' else "TimeStop 90m"
                    exit_time_str = t5
                    break

            cost = invested * FEE_RATE
            net_pnl = (shares * (exit_px - entry_px)) - cost
            capital += net_pnl
            ret = net_pnl / invested

            trades.append({
                "date": d_str,
                "symbol": chosen_sym,
                "entry_time": cur_time,
                "entry_t": time_str,
                "exit_t": exit_time_str,
                "entry_px": entry_px,
                "exit_px": exit_px,
                "ret_pct": round(ret * 100, 2),
                "net_pnl": round(net_pnl, 2),
                "exit_reason": exit_reason,
                "bars_held": bars_held,
                "holding_minutes": bars_held * 5
            })

            consumed = int(np.ceil(bars_held / 3.0))
            b_idx += max(1, consumed)

    tdf = pd.DataFrame(trades)
    print(f"\n========================================================")
    print(f"📊 [15분봉 챔피언 48건 매매의 청산 시간대별 정밀 분석]")
    print(f"========================================================")
    print(f"총 거래 수: {len(tdf)}건 | 승: {(tdf['net_pnl']>0).sum()}건 | 패: {(tdf['net_pnl']<=0).sum()}건 | 승률: {(tdf['net_pnl']>0).mean()*100:.1f}%")

    def group_exit(t_str):
        if t_str < '12:00': return '1. 오전 청산 (09:30~12:00)'
        elif t_str < '14:30': return '2. 점심/오후 청산 (12:00~14:30)'
        elif t_str < '15:00': return '3. 마감 전반 청산 (14:30~15:00)'
        elif t_str < '15:30': return '4. 파워아워 전반 (15:00~15:30)'
        else: return '5. 장 마감 직전 (15:30~15:50 EOD)'

    tdf['exit_group'] = tdf['exit_t'].apply(group_exit)

    summary = tdf.groupby('exit_group').agg(
        trades=('ret_pct', 'count'),
        wins=('ret_pct', lambda x: (x > 0).sum()),
        win_rate=('ret_pct', lambda x: (x > 0).mean() * 100),
        avg_ret=('ret_pct', 'mean'),
        total_pnl=('net_pnl', 'sum'),
        tp_count=('exit_reason', lambda x: (x == 'TP +3.0%').sum()),
        sl_count=('exit_reason', lambda x: (x == 'SL -2.0%').sum()),
        eod_count=('exit_reason', lambda x: x.str.contains('EOD|TimeStop').sum()),
        avg_hold_min=('holding_minutes', 'mean')
    ).reset_index()

    print("\n--- [청산 시간대별 성과 요약표] ---")
    print(summary.to_string(index=False))

    print("\n--- [파워아워/장 마감 (15:00 이후)까지 포지션을 보유했던 거래 전수 내역] ---")
    late = tdf[tdf['exit_t'] >= '15:00']
    print(f"총 {len(late)}건:")
    print(late[['date', 'symbol', 'entry_t', 'exit_t', 'entry_px', 'exit_px', 'ret_pct', 'exit_reason', 'holding_minutes']].to_string(index=False))

if __name__ == '__main__':
    analyze_close_riding()
