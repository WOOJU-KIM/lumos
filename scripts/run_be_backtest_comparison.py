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

def run_be_backtest_comparison():
    lake = MarketDataLake()
    print("⏳ 데이터 로드 및 15분봉 챔피언 백테스트 준비 중...")
    
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

    # Simulation engine parameterized for:
    # - max_entry_time: '14:30' or '14:15'
    # - enable_be_stop: bool
    # - be_trigger_pct: 0.010 (+1.0%)
    # - be_sl_pct: 0.005 (+0.5%)
    def simulate_mode(max_entry_time: str, enable_be_stop: bool, be_trigger_pct: float = 0.010, be_sl_pct: float = 0.005):
        capital = 10000.0
        trades = []
        equity_curve = [capital]
        daily_stoploss_count = 0

        for d_str in unique_dates:
            day_15 = long_15m_feat[long_15m_feat['date_str'] == d_str]
            day_5_l = long_5m[long_5m['date_str'] == d_str]
            day_5_s = short_5m[short_5m['date_str'] == d_str]
            if len(day_15) < 5: continue

            b_idx = 0
            n_bars = len(day_15)
            daily_stoploss_count = 0

            while b_idx < n_bars:
                row_15 = day_15.iloc[b_idx]
                cur_time = row_15['datetime']
                time_str = row_15['time_str']

                if b_idx < 1 or time_str > max_entry_time or daily_stoploss_count >= 3:
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
                default_sl_px = round(entry_px * (1 + SL_15M), 2)
                exit_px = entry_px
                exit_reason = "TimeStop"
                exit_time_str = ""
                bars_held = 0
                max_high_reached = entry_px

                eval_5m = post_5m.iloc[:TIME_STOP_BARS]
                for k in range(len(eval_5m)):
                    c5 = eval_5m.iloc[k]
                    t5 = c5['time_str']
                    c5_o, c5_h, c5_l, c5_c = float(c5['Open']), float(c5['High']), float(c5['Low']), float(c5['Close'])
                    bars_held = k + 1
                    max_high_reached = max(max_high_reached, c5_h)

                    # Dynamic SL calculation
                    current_sl_px = default_sl_px
                    is_be_active = False
                    if enable_be_stop and t5 >= '15:00':
                        # ONLY if max profit achieved so far was >= +1.0%
                        if (max_high_reached - entry_px) / entry_px >= be_trigger_pct:
                            current_sl_px = round(entry_px * (1 + be_sl_pct), 2)
                            is_be_active = True

                    hit_tp = (c5_h >= tp_px)
                    hit_sl = (c5_l <= current_sl_px)

                    if hit_tp and hit_sl:
                        if c5_c >= c5_o:
                            exit_px = round(tp_px - SLIPPAGE, 2); exit_reason = "TP +3.0%"
                        else:
                            exit_px = round(current_sl_px - SLIPPAGE, 2)
                            exit_reason = "BE +0.5%" if is_be_active else "SL -2.0%"
                            if not is_be_active: daily_stoploss_count += 1
                        exit_time_str = t5
                        break
                    elif hit_tp:
                        exit_px = round(tp_px - SLIPPAGE, 2); exit_reason = "TP +3.0%"
                        exit_time_str = t5
                        break
                    elif hit_sl:
                        exit_px = round(current_sl_px - SLIPPAGE, 2)
                        exit_reason = "BE +0.5%" if is_be_active else "SL -2.0%"
                        if not is_be_active: daily_stoploss_count += 1
                        exit_time_str = t5
                        break
                    elif t5 >= '15:45' or k == len(eval_5m) - 1:
                        exit_px = round(c5_c - SLIPPAGE, 2)
                        exit_reason = "15:50 EOD" if t5 >= '15:45' else "TimeStop 90m"
                        exit_time_str = t5
                        if (exit_px - entry_px) / entry_px <= -0.020:
                            daily_stoploss_count += 1
                        break

                cost = invested * FEE_RATE
                net_pnl = (shares * (exit_px - entry_px)) - cost
                capital += net_pnl
                equity_curve.append(capital)
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

        tot_tr = len(trades)
        w_tr = [t for t in trades if t['net_pnl'] > 0]
        l_tr = [t for t in trades if t['net_pnl'] <= 0]
        wr = (len(w_tr) / tot_tr * 100) if tot_tr > 0 else 0.0
        gross_win = sum([t['ret_pct'] for t in w_tr])
        gross_loss = abs(sum([t['ret_pct'] for t in l_tr])) if l_tr else 0.0001
        pf = (gross_win / gross_loss) if gross_loss > 0 else 99.9
        cum_ret = ((capital - 10000.0) / 10000.0) * 100.0

        eq_arr = np.array(equity_curve)
        pk = np.maximum.accumulate(eq_arr)
        dd = (pk - eq_arr) / pk * 100.0
        mdd = np.max(dd) if len(dd) > 0 else 0.0

        return {
            "total_trades": tot_tr,
            "wins": len(w_tr),
            "losses": len(l_tr),
            "win_rate": wr,
            "profit_factor": pf,
            "cum_return": cum_ret,
            "mdd": mdd,
            "final_capital": capital,
            "trades": trades
        }

    # 1. Base: 현재 상태 (14:30 컷오프, BE 스탑 없음)
    print("⏳ [1/3] 기준 모델 (현재 상태: 14:30 컷오프, BE 없음) 백테스트...")
    base_res = simulate_mode(max_entry_time='14:30', enable_be_stop=False)

    # 2. Case 1: 1번만 적용 (14:30 컷오프 유지 + 15:00 BE +0.5% 스탑)
    print("⏳ [2/3] Case 1 (1번만 적용: 14:30 유지 + 15:00 BE +0.5% 스탑) 백테스트...")
    case1_res = simulate_mode(max_entry_time='14:30', enable_be_stop=True, be_trigger_pct=0.010, be_sl_pct=0.005)

    # 3. Case 2: 1번+2번 둘 다 적용 (14:15 컷오프 + 15:00 BE +0.5% 스탑)
    print("⏳ [3/3] Case 2 (1번+2번 적용: 14:15 단축 + 15:00 BE +0.5% 스탑) 백테스트...")
    case2_res = simulate_mode(max_entry_time='14:15', enable_be_stop=True, be_trigger_pct=0.010, be_sl_pct=0.005)

    print("\n" + "="*95)
    print("📊 [비교 성적표: 기준(현재) vs Case 1(1번만) vs Case 2(1+2번 둘 다)]")
    print("="*95)

    df_comp = pd.DataFrame([
        {
            "시나리오": "현재 기준 (Base)",
            "진입 컷오프": "14:30",
            "15시 BE 스탑": "미적용",
            "총 거래": f"{base_res['total_trades']}회",
            "승 / 패": f"{base_res['wins']}승 {base_res['losses']}패",
            "승률(%)": f"{base_res['win_rate']:.1f}%",
            "손익비(PF)": f"{base_res['profit_factor']:.2f}",
            "누적수익률": f"{base_res['cum_return']:+.2f}%",
            "MDD(%)": f"{base_res['mdd']:.2f}%",
            "최종자본금": f"${base_res['final_capital']:,.2f}"
        },
        {
            "시나리오": "Case 1 (1번만 적용)",
            "진입 컷오프": "14:30 유지",
            "15시 BE 스탑": "+1% 도달 시 +0.5% 상향",
            "총 거래": f"{case1_res['total_trades']}회",
            "승 / 패": f"{case1_res['wins']}승 {case1_res['losses']}패",
            "승률(%)": f"{case1_res['win_rate']:.1f}%",
            "손익비(PF)": f"{case1_res['profit_factor']:.2f}",
            "누적수익률": f"{case1_res['cum_return']:+.2f}%",
            "MDD(%)": f"{case1_res['mdd']:.2f}%",
            "최종자본금": f"${case1_res['final_capital']:,.2f}"
        },
        {
            "시나리오": "Case 2 (1+2번 둘 다)",
            "진입 컷오프": "14:15로 15분 단축",
            "15시 BE 스탑": "+1% 도달 시 +0.5% 상향",
            "총 거래": f"{case2_res['total_trades']}회",
            "승 / 패": f"{case2_res['wins']}승 {case2_res['losses']}패",
            "승률(%)": f"{case2_res['win_rate']:.1f}%",
            "손익비(PF)": f"{case2_res['profit_factor']:.2f}",
            "누적수익률": f"{case2_res['cum_return']:+.2f}%",
            "MDD(%)": f"{case2_res['mdd']:.2f}%",
            "최종자본금": f"${case2_res['final_capital']:,.2f}"
        }
    ])
    print(df_comp.to_string(index=False))

    # Detailed trade comparison around changed trades
    print("\n--- [Case 1에서 실제로 방어된 거래 상세] ---")
    t1 = pd.DataFrame(case1_res['trades'])
    tb = pd.DataFrame(base_res['trades'])
    
    diff_idx = []
    for i in range(len(t1)):
        if t1.iloc[i]['exit_reason'] != tb.iloc[i]['exit_reason'] or t1.iloc[i]['ret_pct'] != tb.iloc[i]['ret_pct']:
            diff_idx.append(i)
            
    if diff_idx:
        for idx in diff_idx:
            print(f"📌 날짜: {t1.iloc[idx]['date']} | 종목: {t1.iloc[idx]['symbol']} | 진입: {t1.iloc[idx]['entry_t']} @ ${t1.iloc[idx]['entry_px']}")
            print(f"   [기존 Base]: 청산 {tb.iloc[idx]['exit_reason']} | 손익 {tb.iloc[idx]['ret_pct']:+.2f}%")
            print(f"   [Case 1   ]: 청산 {t1.iloc[idx]['exit_reason']} | 손익 {t1.iloc[idx]['ret_pct']:+.2f}%")
    else:
        print("변경된 거래 없음")

    print("\n--- [Case 2에서 14:15 단축으로 제외된 거래 상세] ---")
    t2 = pd.DataFrame(case2_res['trades'])
    print(f"Case 1 거래 수({len(t1)}) vs Case 2 거래 수({len(t2)}) -> 차이 {len(t1) - len(t2)}건 제외됨")
    dates_in_1_not_2 = set(t1['date']) - set(t2['date'])
    for d in dates_in_1_not_2:
        row = t1[t1['date'] == d].iloc[0]
        print(f"❌ 제외된 거래: 날짜 {d} | 종목 {row['symbol']} | 진입 {row['entry_t']} | 손익 {row['ret_pct']:+.2f}% ({row['exit_reason']})")

if __name__ == '__main__':
    run_be_backtest_comparison()
