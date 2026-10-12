import os
import sys
import json
from pathlib import Path
import config
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(os.getcwd())
sys.path.insert(0, str(PROJECT_ROOT))

from core.data_lake import MarketDataLake
from core.ml_engine import MLFeatureEngine

def sanitize_for_json(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)): return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)): return float(obj)
    elif isinstance(obj, dict): return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [sanitize_for_json(i) for i in obj]
    return obj

def mk_ca(df, name, periods=[1, 3, 5, 20]):
    d = df.copy()
    d['datetime_dt'] = pd.to_datetime(d['datetime']) + pd.Timedelta(minutes=15)
    for p in periods:
        d[f'{name}_ret_{p}'] = d['Close'].pct_change(p) * 100.0
    d[f'{name}_dir_1']    = np.sign(d['Close'].pct_change(1))
    d[f'{name}_momentum'] = d['Close'].pct_change(1) - d['Close'].pct_change(5) / 5.0
    return d[['datetime_dt'] + [c for c in d.columns if c.startswith(name)]]

def simulate_with_threshold(test_df, long_5m_by_date, short_5m_by_date, short_dict,
                             conf_threshold, initial_capital=10_000_000.0):
    """주어진 확신도 임계값으로 5분봉 체결 시뮬레이션"""
    current_capital  = initial_capital
    peak_capital     = initial_capital
    max_drawdown_pct = 0.0
    trades = []; trade_id = 0; slippage_rate = 0.0020

    for d_str in sorted(test_df['date_str'].unique()):
        day_bars_15 = test_df[test_df['date_str'] == d_str].reset_index(drop=True)
        day_long_5  = long_5m_by_date.get(d_str, pd.DataFrame())
        day_short_5  = short_5m_by_date.get(d_str, pd.DataFrame())
        daily_stoploss_count = 0; b_idx = 0

        while b_idx < len(day_bars_15):
            row = day_bars_15.iloc[b_idx]
            curr_dt = row['datetime']; time_str = row['time_str']
            cur_long_close = float(row['Close'])
            short_row = short_dict.get(curr_dt)
            cur_short_close = float(short_row['Close']) if short_row else 0.0

            if daily_stoploss_count >= 3 or time_str >= '14:30':
                b_idx += 1; continue

            gbdt_dir  = row.get('Direction', 'NONE')
            gbdt_conf = float(row.get('Confidence', 0.50))
            entry_approved = False; target_sym = None; entry_price = 0.0

            if gbdt_dir == "LONG" and gbdt_conf >= conf_threshold:
                entry_approved = True; target_sym = config.BASE_ASSET_LONG; entry_price = cur_long_close
            elif gbdt_dir == "SHORT" and gbdt_conf >= conf_threshold:
                entry_approved = True; target_sym = config.BASE_ASSET_SHORT; entry_price = cur_short_close

            if entry_approved and entry_price > 0:
                target_5m = day_long_5 if target_sym == config.BASE_ASSET_LONG else day_short_5
                post_5m   = target_5m[target_5m['datetime'] > curr_dt].reset_index(drop=True)

                atr_pct             = float(row.get('ATR_Pct', 1.8))
                sl_pct              = max(0.020, min(0.032, (atr_pct / 100.0) * 1.5))
                tp_max_px           = entry_price * 1.035
                trailing_trigger_px = entry_price * 1.020
                current_sl_px       = entry_price * (1.0 - sl_pct)
                is_trailing_active  = False; peak_high = entry_price
                exit_price = None; exit_reason = None; exit_dt = None; holding_5m_bars = 0

                for k in range(min(len(post_5m), 18)):
                    b = post_5m.iloc[k]
                    b_h = float(b['High']); b_l = float(b['Low'])
                    b_c = float(b['Close']); b_dt = b['datetime']; b_time = b_dt[11:16]
                    if b_h > peak_high: peak_high = b_h
                    if not is_trailing_active and peak_high >= trailing_trigger_px:
                        is_trailing_active = True; current_sl_px = entry_price * 1.002
                    if is_trailing_active:
                        t = peak_high * 0.992
                        if t > current_sl_px: current_sl_px = t
                    if b_h >= tp_max_px:
                        exit_price = tp_max_px; exit_reason = "TP_MAX"
                        exit_dt = b_dt; holding_5m_bars = k + 1; break
                    if b_l <= current_sl_px:
                        exit_price = current_sl_px
                        exit_reason = "TRAIL_SL" if is_trailing_active else "ATR_SL"
                        exit_dt = b_dt; holding_5m_bars = k + 1; break
                    if b_time >= '15:45':
                        exit_price = b_c; exit_reason = "EOD"
                        exit_dt = b_dt; holding_5m_bars = k + 1; break
                    if k == 17:
                        exit_price = b_c; exit_reason = "TIMESTOP"
                        exit_dt = b_dt; holding_5m_bars = 18; break

                if exit_price is None:
                    exit_price = float(post_5m.iloc[-1]['Close']) if not post_5m.empty else entry_price
                    exit_reason = "EOD"; exit_dt = post_5m.iloc[-1]['datetime'] if not post_5m.empty else curr_dt
                    holding_5m_bars = len(post_5m) if not post_5m.empty else 1

                raw_ret = (exit_price / entry_price) - 1.0
                net_ret = raw_ret - slippage_rate
                pnl_krw = current_capital * net_ret
                current_capital += pnl_krw
                if net_ret <= -0.015: daily_stoploss_count += 1
                if current_capital > peak_capital: peak_capital = current_capital
                dd = (peak_capital - current_capital) / peak_capital * 100.0
                if dd > max_drawdown_pct: max_drawdown_pct = dd
                trade_id += 1
                trades.append({
                    'trade_id': trade_id, 'symbol': target_sym,
                    'entry_dt': curr_dt, 'exit_dt': exit_dt,
                    'net_ret_pct': round(net_ret * 100, 2),
                    'pnl_krw': int(round(pnl_krw)),
                    'ending_capital': int(round(current_capital)),
                    'is_win': 1 if net_ret > 0 else 0,
                    'conf': round(gbdt_conf, 3)
                })
                b_idx += max(1, int(np.ceil(holding_5m_bars / 3.0)))
            else:
                b_idx += 1

    return trades, int(current_capital), round(max_drawdown_pct, 2)

def print_yearly(df_t, label, initial_capital):
    df_t['year']  = df_t['entry_dt'].str.slice(0, 4)
    df_t['month'] = df_t['entry_dt'].str.slice(0, 7)
    print(f"\n  [{label}]")
    print(f"  {'연도':<5} | {'시작 잔고':>15} | {'기말 잔고':>15} | {'수익률':>8} | {'거래수':>5} | {'승률':>7} | {'PF':>5} | TQQQ/SQQQ")
    print("  " + "-" * 90)
    y_start_cap = initial_capital
    yearly = []
    for y, g in df_t.groupby('year'):
        y_end_cap = float(g['ending_capital'].iloc[-1]); y_ret = (y_end_cap / y_start_cap - 1.0) * 100.0
        y_trades = len(g); w_cnt = len(g[g['net_ret_pct'] > 0])
        y_wr = (w_cnt / y_trades * 100.0) if y_trades > 0 else 0.0
        gw = g[g['net_ret_pct'] > 0]['net_ret_pct'].sum()
        gl = abs(g[g['net_ret_pct'] <= 0]['net_ret_pct'].sum())
        y_pf = (gw / gl) if gl > 0 else 99.0
        long_c = len(g[g['symbol'] == config.BASE_ASSET_LONG]); short_c = len(g[g['symbol'] == config.BASE_ASSET_SHORT])
        print(f"  {y:<5} | {int(y_start_cap):>13,}원 | {int(y_end_cap):>13,}원 | {y_ret:>+7.1f}% | {y_trades:>4}회 | {y_wr:>6.1f}% | {y_pf:>5.2f} | {long_c}/{short_c}회")
        yearly.append({"year": y, "start_cap": int(y_start_cap), "end_cap": int(y_end_cap),
                        "ret_pct": round(y_ret,1), "trades": y_trades, "win_rate": round(y_wr,1),
                        "pf": round(y_pf,2), "long_cnt": long_c, "short_cnt": short_c})
        y_start_cap = y_end_cap
    return yearly

def run_confidence_sweep():
    lake = MarketDataLake()
    CONF_THRESHOLDS = [0.60, 0.62, 0.65, 0.68]
    print("=" * 115)
    print("🚀 [Lumos V4 확신도 임계값 스윕: 60% / 62% / 65% / 68% 동시 비교]")
    print("   • GBDT 학습은 1회 (V4 동일 구조), 확신도 임계값만 4가지로 변경")
    print("   • 기대: 높은 임계값 → 거래 횟수 감소, 하지만 진입 품질 향상")
    print("=" * 115)

    # 1. 데이터 로드
    print("⏳ [1/4] 데이터 로드 중...")
    long_15m = lake.load_candles(config.BASE_ASSET_LONG, "15m").sort_values('datetime').reset_index(drop=True)
    short_15m = lake.load_candles(config.BASE_ASSET_SHORT, "15m").sort_values('datetime').reset_index(drop=True)
    macro_60m = lake.load_candles(config.MACRO_TREND_SYMBOL, "60m").sort_values('datetime').reset_index(drop=True)
    nvda_15m  = lake.load_candles("NVDA",  "15m").sort_values('datetime').reset_index(drop=True)
    qqq_15m   = lake.load_candles("QQQ",   "15m").sort_values('datetime').reset_index(drop=True)
    vixy_15m  = lake.load_candles("VIXY",  "15m").sort_values('datetime').reset_index(drop=True)
    ief_15m   = lake.load_candles("IEF",   "15m").sort_values('datetime').reset_index(drop=True)
    long_5m   = lake.load_candles(config.BASE_ASSET_LONG, "5m").sort_values('datetime').reset_index(drop=True)
    short_5m   = lake.load_candles(config.BASE_ASSET_SHORT, "5m").sort_values('datetime').reset_index(drop=True)

    # 2. 피처 생성 (V4 동일)
    print("⏳ [2/4] V4 피처 생성 중...")
    macro_60m['ema20'] = macro_60m['Close'].ewm(span=20, adjust=False).mean()
    macro_60m['ema60'] = macro_60m['Close'].ewm(span=60, adjust=False).mean()
    macro_60m['macro_trend_spread']   = (macro_60m['ema20'] - macro_60m['ema60']) / (macro_60m['ema60'] + 1e-9) * 100.0
    macro_60m['macro_ema20_slope']    = (macro_60m['ema20'].diff(5) / macro_60m['ema20'].shift(5)) * 100.0
    macro_60m['macro_price_vs_ema20'] = (macro_60m['Close'] - macro_60m['ema20']) / (macro_60m['ema20'] + 1e-9) * 100.0
    macro_cols = ['macro_trend_spread', 'macro_ema20_slope', 'macro_price_vs_ema20']
    macro_feat = macro_60m[['datetime'] + macro_cols].copy()
    macro_feat['datetime_dt'] = pd.to_datetime(macro_feat['datetime']) + pd.Timedelta(minutes=60)

    ml_engine = MLFeatureEngine(confidence_threshold=0.60)
    df_15m = ml_engine.extract_features(long_15m)
    df_15m['ATR_Pct']         = (df_15m['ATR_14']     / (df_15m['Close'] + 1e-9)) * 100.0
    df_15m['MACD_Pct']        = (df_15m['MACD']        / (df_15m['Close'] + 1e-9)) * 100.0
    df_15m['MACD_Signal_Pct'] = (df_15m['MACD_Signal'] / (df_15m['Close'] + 1e-9)) * 100.0
    df_15m['MACD_Hist_Pct']   = (df_15m['MACD_Hist']   / (df_15m['Close'] + 1e-9)) * 100.0
    df_15m['datetime_dt']     = pd.to_datetime(df_15m['datetime']) + pd.Timedelta(minutes=15)

    for feat_df in [mk_ca(nvda_15m,'nvda'), mk_ca(qqq_15m,'qqq'),
                    mk_ca(vixy_15m,'vixy'), mk_ca(ief_15m,'ief')]:
        df_merged_base = feat_df if 'df_merged_base' not in dir() else df_merged_base
    df_merged = pd.merge_asof(df_15m.sort_values('datetime_dt'),
                               macro_feat.sort_values('datetime_dt')[['datetime_dt']+macro_cols],
                               on='datetime_dt', direction='backward')
    for sym, raw in [('nvda', nvda_15m), ('qqq', qqq_15m), ('vixy', vixy_15m), ('ief', ief_15m)]:
        df_merged = pd.merge_asof(df_merged.sort_values('datetime_dt'),
                                   mk_ca(raw, sym).sort_values('datetime_dt'),
                                   on='datetime_dt', direction='backward')

    df_merged['long_ret_20']    = df_merged['Close'].pct_change(20) * 100.0
    df_merged['long_ret_5']     = df_merged['Close'].pct_change(5)  * 100.0
    df_merged['long_vs_qqq_20'] = df_merged['long_ret_20'] - df_merged.get('qqq_ret_20', pd.Series(0.0, index=df_merged.index)).fillna(0)
    df_merged['long_vs_qqq_5']  = df_merged['long_ret_5']  - df_merged.get('qqq_ret_5',  pd.Series(0.0, index=df_merged.index)).fillna(0)
    df_merged['panic_signal']   = (
        (df_merged.get('vixy_ret_1', pd.Series(0.0, index=df_merged.index)).fillna(0) > 0).astype(int) +
        (df_merged.get('ief_ret_1',  pd.Series(0.0, index=df_merged.index)).fillna(0) > 0).astype(int)
    )
    labels = ml_engine.compute_triple_barrier_labels(df_merged)
    df_merged['target']   = labels.map({1: 2, -1: 0, 0: 1}).fillna(1).astype(int)
    df_merged['date_str'] = df_merged['datetime_dt'].dt.strftime('%Y-%m-%d')
    df_merged['time_str'] = df_merged['datetime_dt'].dt.strftime('%H:%M')
    df_merged['week_id']  = (df_merged['datetime_dt'].dt.isocalendar().year.astype(str) + '-' +
                              df_merged['datetime_dt'].dt.isocalendar().week.astype(str).str.zfill(2))

    raw_price_features = {
        'open','high','low','close','volume','Open','High','Low','Close','Volume',
        'datetime','datetime_dt','date_str','time_str','week_id','target','date','year',
        'cum_vp','vwap','EMA_9','EMA_21','EMA_50','EMA_200',
        'BB_Upper','BB_Lower','KC_Upper','KC_Lower','ATR_14','MACD','MACD_Signal','MACD_Hist'
    }
    feature_cols = [c for c in df_merged.columns
                    if c not in raw_price_features and pd.api.types.is_numeric_dtype(df_merged[c])]
    print(f"   ✅ 피처 {len(feature_cols)}개 (V4 동일)")

    # 3. WFA 추론 (1회만)
    unique_weeks = sorted(df_merged['week_id'].unique())
    ROLLING_WINDOW_WEEKS = 104
    print(f"⏳ [3/4] GBDT 주간 롤링 WFA 추론 (1회, 총 {len(unique_weeks) - ROLLING_WINDOW_WEEKS}주)...")

    test_dfs = []
    for w_idx in range(ROLLING_WINDOW_WEEKS, len(unique_weeks)):
        cur_test_week = unique_weeks[w_idx]
        train_weeks   = unique_weeks[w_idx - ROLLING_WINDOW_WEEKS : w_idx]
        X_tr = df_merged.loc[df_merged['week_id'].isin(train_weeks), feature_cols].fillna(0.0)
        y_tr = df_merged.loc[df_merged['week_id'].isin(train_weeks), 'target']
        clf = LGBMClassifier(objective='multiclass', num_class=3, class_weight='balanced',
                              n_estimators=85, max_depth=4, learning_rate=0.03,
                              random_state=42, verbosity=-1, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        w_test_df = df_merged[df_merged['week_id'] == cur_test_week].copy()
        if w_test_df.empty: continue
        w_probs = clf.predict_proba(w_test_df[feature_cols].fillna(0.0))
        p_s, p_n, p_l = w_probs[:, 0], w_probs[:, 1], w_probs[:, 2]
        w_confs = np.full(len(w_test_df), 0.50, dtype=float)
        w_dirs  = ["NONE"] * len(w_test_df)
        for i in range(len(w_test_df)):
            ps, pn, pl = p_s[i], p_n[i], p_l[i]
            if pl > pn and pl > ps:
                w_confs[i] = min(0.95, max(0.50, 0.50 + (pl - 0.333) * 1.15))
                w_dirs[i]  = "LONG"
            elif ps > pn and ps > pl:
                w_confs[i] = min(0.95, max(0.50, 0.50 + (ps - 0.333) * 1.15))
                w_dirs[i]  = "SHORT"
        w_test_df['Confidence'] = w_confs
        w_test_df['Direction']  = w_dirs
        test_dfs.append(w_test_df)

    test_df = pd.concat(test_dfs, ignore_index=True)
    print("   ✅ GBDT WFA 추론 완료 → 이제 4개 임계값으로 시뮬레이션")

    # 4. 5분봉 데이터 준비
    long_5m_by_date = {d: g.sort_values('datetime').reset_index(drop=True)
                        for d, g in long_5m.groupby(long_5m['datetime'].str.slice(0, 10))}
    short_5m_by_date = {d: g.sort_values('datetime').reset_index(drop=True)
                        for d, g in short_5m.groupby(short_5m['datetime'].str.slice(0, 10))}
    short_dict = short_15m.set_index('datetime').to_dict(orient='index')

    # 5. 4개 임계값 동시 시뮬레이션
    print("\n⏳ [4/4] 4개 확신도 임계값 시뮬레이션 중...")
    initial_capital = 10_000_000.0
    all_results = {}

    for conf_th in CONF_THRESHOLDS:
        label = f"conf_{int(conf_th*100)}pct"
        print(f"   🔄 임계값 {conf_th:.0%} 시뮬레이션...")
        trades, final_cap, mdd = simulate_with_threshold(
            test_df, long_5m_by_date, short_5m_by_date, short_dict,
            conf_th, initial_capital
        )
        all_results[label] = {'conf_threshold': conf_th, 'trades': trades,
                               'final_capital': final_cap, 'mdd_pct': mdd}
        print(f"      → 거래: {len(trades)}회 | 최종: {final_cap:,}원 | MDD: {mdd:.1f}%")

    # 6. 비교 결과 출력
    print("\n" + "=" * 115)
    print("📊 [확신도 임계값 비교 결과 요약]")
    print("=" * 115)
    print(f"{'임계값':>8} | {'최종 잔고':>18} | {'총 수익률':>10} | {'거래수':>6} | {'승률':>8} | {'PF':>6} | {'MDD':>8}")
    print("-" * 80)

    summary_all = {}
    for conf_th in CONF_THRESHOLDS:
        label = f"conf_{int(conf_th*100)}pct"
        r = all_results[label]
        df_t = pd.DataFrame(r['trades'])
        if df_t.empty: continue
        total_ret = (r['final_capital'] / initial_capital - 1.0) * 100.0
        w_cnt = len(df_t[df_t['net_ret_pct'] > 0]); tot = len(df_t)
        wr = (w_cnt / tot * 100.0) if tot > 0 else 0.0
        gw = df_t[df_t['net_ret_pct'] > 0]['net_ret_pct'].sum()
        gl = abs(df_t[df_t['net_ret_pct'] <= 0]['net_ret_pct'].sum())
        pf = (gw / gl) if gl > 0 else 99.0
        print(f"  {conf_th:.0%}    | {r['final_capital']:>16,}원 | {total_ret:>+9.1f}% | {tot:>5}회 | {wr:>7.1f}% | {pf:>6.2f} | {r['mdd_pct']:>7.1f}%")
        summary_all[label] = {'conf_threshold': conf_th, 'final_capital': r['final_capital'],
                               'total_ret_pct': round(total_ret,1), 'total_trades': tot,
                               'win_rate': round(wr,1), 'pf': round(pf,2), 'mdd_pct': r['mdd_pct']}

    # 7. 연도별 상세 비교
    print("\n" + "=" * 115)
    print("📈 [연도별 상세 비교]")
    for conf_th in CONF_THRESHOLDS:
        label = f"conf_{int(conf_th*100)}pct"
        df_t = pd.DataFrame(all_results[label]['trades'])
        if df_t.empty: continue
        yearly = print_yearly(df_t, f"확신도 {conf_th:.0%}", initial_capital)
        summary_all[label]['yearly'] = yearly

    # 저장
    out_file = PROJECT_ROOT / "data" / "backtest_confidence_sweep.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(summary_all), f, indent=2, ensure_ascii=False)
    print(f"\n💾 [결과 저장 완료] {out_file}")

if __name__ == '__main__':
    run_confidence_sweep()
