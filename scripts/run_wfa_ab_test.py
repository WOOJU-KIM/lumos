import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from pathlib import Path
import itertools
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))
from core.data_lake import MarketDataLake
from core.ml_engine import MLFeatureEngine
from core.heterogeneous_models import CrossAssetDislocationModel

def run_wfa_ab_test():
    print("=" * 80)
    print("🚀 [방어막 해제 vs 방어막 장착 주간 롤링 WFA 비교 테스트 - QQQ 60m 패치]")
    print("=" * 80)

    lake = MarketDataLake()
    tqqq_15m = lake.load_candles('TQQQ', '15m')
    sqqq_15m = lake.load_candles('SQQQ', '15m')
    qqq_60m = lake.load_candles('QQQ', '60m')
    nvda_15m = lake.load_candles('NVDA', '15m')
    soxx_15m = lake.load_candles('SOXX', '15m')
    qqq_15m  = lake.load_candles('QQQ', '15m')
    vixy_15m = lake.load_candles('VIXY', '15m')
    ief_15m  = lake.load_candles('IEF', '15m')
    tqqq_5m = lake.load_candles('TQQQ', '5m')
    sqqq_5m = lake.load_candles('SQQQ', '5m')
    
    qqq_60m['ema20'] = qqq_60m['Close'].ewm(span=20, adjust=False).mean()
    qqq_60_dt = pd.to_datetime(qqq_60m['datetime'])
    qqq_60m['dt_hr'] = qqq_60_dt.dt.strftime('%Y-%m-%d %H')
    ema_map = qqq_60m.set_index('dt_hr')['ema20'].to_dict()
    
    tqqq_5m_by_date = {}
    for d, g in tqqq_5m.groupby(tqqq_5m['datetime'].str.slice(0, 10)):
        tqqq_5m_by_date[d] = g.sort_values('datetime').reset_index(drop=True)
    sqqq_5m_by_date = {}
    for d, g in sqqq_5m.groupby(sqqq_5m['datetime'].str.slice(0, 10)):
        sqqq_5m_by_date[d] = g.sort_values('datetime').reset_index(drop=True)
        
    ml_engine = MLFeatureEngine(confidence_threshold=0.60)
    tqqq_feat = ml_engine.extract_features(tqqq_15m)
    labels = ml_engine.compute_triple_barrier_labels(tqqq_feat)
    tqqq_feat['target'] = labels.map({1: 2, -1: 0, 0: 1}).fillna(1).astype(int)
    
    tqqq_feat['datetime_dt'] = pd.to_datetime(tqqq_feat['datetime'])
    tqqq_feat['date_str'] = tqqq_feat['datetime_dt'].dt.strftime('%Y-%m-%d')
    tqqq_feat['week_id'] = tqqq_feat['datetime_dt'].dt.isocalendar().year.astype(str) + '-' + tqqq_feat['datetime_dt'].dt.isocalendar().week.astype(str).str.zfill(2)
    
    raw_price_features = {'open', 'high', 'low', 'close', 'volume', 'Open', 'High', 'Low', 'Close', 'Volume',
        'datetime', 'datetime_dt', 'date_str', 'time_str', 'week_id', 'target', 'date', 'year'}
    feature_cols = [c for c in tqqq_feat.columns if c not in raw_price_features and pd.api.types.is_numeric_dtype(tqqq_feat[c])]
    
    nvda_map = nvda_15m.set_index('datetime')['Close'].to_dict()
    soxx_map = soxx_15m.set_index('datetime')['Close'].to_dict()
    qqq_map  = qqq_15m.set_index('datetime')['Close'].to_dict()
    vix_map  = vixy_15m.set_index('datetime')['Close'].to_dict()
    ief_map  = ief_15m.set_index('datetime')['Close'].to_dict()
    
    cross_mod = CrossAssetDislocationModel(dislocation_z_threshold=1.6)
    unique_weeks = sorted(tqqq_feat['week_id'].unique())
    ROLLING_WINDOW_WEEKS = 104
    
    print("▶ 1단계: 104주 롤링 WFA 추론 진행 중...")
    test_dfs = []
    for w_idx in range(ROLLING_WINDOW_WEEKS, len(unique_weeks)):
        cur_test_week = unique_weeks[w_idx]
        train_weeks = unique_weeks[w_idx - ROLLING_WINDOW_WEEKS : w_idx]
        train_mask = tqqq_feat['week_id'].isin(train_weeks)
        X_tr = tqqq_feat.loc[train_mask, feature_cols].fillna(0.0)
        y_tr = tqqq_feat.loc[train_mask, 'target']

        clf = LGBMClassifier(
            objective='multiclass', num_class=3, class_weight='balanced',
            n_estimators=80, max_depth=4, learning_rate=0.03, random_state=42, verbosity=-1, n_jobs=-1
        )
        clf.fit(X_tr, y_tr)

        cur_w_mask = tqqq_feat['week_id'] == cur_test_week
        w_test_df = tqqq_feat[cur_w_mask].copy()
        if w_test_df.empty: continue

        X_te = w_test_df[feature_cols].fillna(0.0)
        w_probs = clf.predict_proba(X_te)
        
        w_confs = np.full(len(w_test_df), 0.50, dtype=float)
        w_dirs = ['NONE'] * len(w_test_df)
        for i in range(len(w_test_df)):
            ps, pn, pl = w_probs[i, 0], w_probs[i, 1], w_probs[i, 2]
            if pl > pn and pl > ps:
                w_confs[i] = min(0.95, max(0.50, 0.50 + (pl - 0.333) * 1.15))
                w_dirs[i] = 'LONG_TQQQ'
            elif ps > pn and ps > pl:
                w_confs[i] = min(0.95, max(0.50, 0.50 + (ps - 0.333) * 1.15))
                w_dirs[i] = 'SHORT_SQQQ'

        w_test_df['Confidence'] = w_confs
        w_test_df['Direction'] = w_dirs
        test_dfs.append(w_test_df)
        
    test_df = pd.concat(test_dfs, ignore_index=True)
    unique_dates = sorted(test_df['date_str'].unique())
    print("▶ 추론 완료! 2단계: 백테스트 시뮬레이션 시작 (단일 포지션 릴레이 적용)")
    
    def run_simulation(use_safety_nets=False):
        cap = 10000000.0
        max_cap = cap
        mdd = 0.0
        wins, losses = 0, 0
        eq_curve = []
        
        for d_str in unique_dates:
            day_bars = test_df[test_df['date_str'] == d_str].reset_index(drop=True)
            day_tqqq_5 = tqqq_5m_by_date.get(d_str, pd.DataFrame())
            day_sqqq_5 = sqqq_5m_by_date.get(d_str, pd.DataFrame())
            
            b_idx = 0
            n_bars = len(day_bars)
            while b_idx < n_bars:
                if use_safety_nets and b_idx < 5:
                    b_idx += 1
                    continue
                    
                row = day_bars.iloc[b_idx]
                dt_str = row['datetime']
                entry_time = dt_str[11:16]
                if entry_time >= '15:30':
                    b_idx += 1
                    continue
                
                dir_gbdt = row['Direction']
                conf_gbdt = row['Confidence']
                if dir_gbdt == 'NONE' or conf_gbdt < 0.60:
                    b_idx += 1
                    continue
                
                entry_approved = True
                target_sym = "TQQQ" if dir_gbdt == "LONG_TQQQ" else "SQQQ"
                
                if use_safety_nets:
                    cur_tqqq_close = float(row.get('Close', 1.0))
                    
                    cross_dir = "NONE"
                    n_px = nvda_map.get(dt_str)
                    sx_px = soxx_map.get(dt_str)
                    q_px = qqq_map.get(dt_str)
                    v_px = vix_map.get(dt_str)
                    i_px = ief_map.get(dt_str)
                    
                    qqq_60m_bull = True
                    qqq_60m_bear = True
                    if q_px:
                        hr_key = dt_str[:13]
                        qqq_ema = ema_map.get(hr_key, 0)
                        if qqq_ema > 0:
                            if q_px < qqq_ema * 0.998: qqq_60m_bull = False
                            if q_px > qqq_ema * 1.002: qqq_60m_bear = False
                    
                    if n_px and sx_px and q_px:
                        prev_dt = day_bars.iloc[b_idx - 5]['datetime']
                        p_n = nvda_map.get(prev_dt, n_px)
                        p_sx = soxx_map.get(prev_dt, sx_px)
                        p_q = qqq_map.get(prev_dt, q_px)
                        p_v = vix_map.get(prev_dt, v_px) if v_px else None
                        p_i = ief_map.get(prev_dt, i_px) if i_px else None
                        
                        nvda_r = (n_px / p_n - 1.0) if p_n else 0.0
                        soxx_r = (sx_px / p_sx - 1.0) if p_sx else 0.0
                        qqq_r  = (q_px / p_q - 1.0) if p_q else 0.0
                        vix_r  = (v_px / p_v - 1.0) if (v_px and p_v) else 0.0
                        ief_r  = (i_px / p_i - 1.0) if (i_px and p_i) else 0.0
                        
                        tqqq_r = (cur_tqqq_close / day_bars.iloc[b_idx - 5]['Close']) - 1.0
                        sig_code, _, _ = cross_mod.predict_signal(tqqq_r, nvda_r, soxx_r, qqq_r, vix_r, -ief_r)
                        if sig_code > 0: cross_dir = "LONG_TQQQ"
                        elif sig_code < 0: cross_dir = "SHORT_SQQQ"
                        
                    entry_approved = False
                    if dir_gbdt == "LONG_TQQQ" and qqq_60m_bull and cross_dir != "SHORT_SQQQ":
                        entry_approved = True
                    elif dir_gbdt == "SHORT_SQQQ" and qqq_60m_bear and cross_dir != "LONG_TQQQ":
                        entry_approved = True
                        
                if not entry_approved:
                    b_idx += 1
                    continue
                
                post_5m = day_tqqq_5 if target_sym == 'TQQQ' else day_sqqq_5
                post_5m = post_5m[post_5m['datetime'].str.slice(11, 16) > entry_time]
                if post_5m.empty:
                    b_idx += 1
                    continue
                
                entry_price = float(post_5m.iloc[0].get('Open', post_5m.iloc[0].get('open')))
                atr_14 = float(row.get('ATR_14', 0))
                c_px = float(row.get('Close', 1.0))
                initial_sl = 0.02
                if atr_14 > 0 and c_px > 0:
                    initial_sl = max(0.02, min(0.032, (atr_14 * 1.5) / c_px))
                    
                peak_ret = 0.0; trailing_active = False; exit_ret = 0.0
                exit_time_str = entry_time
                
                for k in range(len(post_5m)):
                    b_5m = post_5m.iloc[k]
                    b_h = float(b_5m.get('High', b_5m.get('high')))
                    b_l = float(b_5m.get('Low', b_5m.get('low')))
                    b_c = float(b_5m.get('Close', b_5m.get('close')))
                    b_time = b_5m['datetime'][11:16]
                    exit_time_str = b_time
                    
                    cur_max_ret = (b_h - entry_price) / entry_price
                    cur_min_ret = (b_l - entry_price) / entry_price
                    cur_c_ret = (b_c - entry_price) / entry_price
                    
                    if cur_max_ret > peak_ret: peak_ret = cur_max_ret
                    if peak_ret >= 0.015: trailing_active = True
                    
                    if cur_min_ret <= -initial_sl:
                        exit_ret = -initial_sl; break
                    elif cur_max_ret >= 0.10:
                        exit_ret = 0.10; break
                    elif trailing_active and (peak_ret - cur_c_ret >= 0.003):
                        exit_ret = cur_c_ret; break
                    if b_time >= '15:50' or k == 23:
                        exit_ret = cur_c_ret; break
                
                net_ret = exit_ret - 0.0020
                if net_ret > 0: wins += 1
                else: losses += 1
                cap *= (1.0 + net_ret)
                if cap > max_cap: max_cap = cap
                dd = (max_cap - cap) / max_cap * 100.0
                if dd > mdd: mdd = dd
                
                while b_idx < n_bars and day_bars.iloc[b_idx]['datetime'][11:16] <= exit_time_str:
                    b_idx += 1
            eq_curve.append({'date': d_str, 'cap': cap})
            
        return wins, losses, mdd, pd.DataFrame(eq_curve)
        
    print("▶ [1/2] 방어막 해제 (순수 GBDT 코어) 백테스트 중...")
    w_off, l_off, mdd_off, df_off = run_simulation(use_safety_nets=False)
    
    print("▶ [2/2] 방어막 장착 (완전체 Lumos + QQQ 60m 적용) 백테스트 중...")
    w_on, l_on, mdd_on, df_on = run_simulation(use_safety_nets=True)
    
    tot_off = w_off + l_off
    wr_off = (w_off / tot_off * 100) if tot_off > 0 else 0
    ret_off = (df_off['cap'].iloc[-1] / 10000000.0 - 1) * 100
    
    tot_on = w_on + l_on
    wr_on = (w_on / tot_on * 100) if tot_on > 0 else 0
    ret_on = (df_on['cap'].iloc[-1] / 10000000.0 - 1) * 100
    
    print("=" * 80)
    print("📊 [비교 백테스트 결과 (WFA - QQQ 60m)]")
    print("--------------------------------------------------------------------------------")
    print(f" 항목            | ⛔ 방어막 해제 (Raw GBDT)     | 🛡️ 방어막 장착 (QQQ 완전체) ")
    print("--------------------------------------------------------------------------------")
    print(f" 총 거래횟수     | {tot_off:>11,}회                 | {tot_on:>11,}회")
    print(f" 승률 (Win Rate) | {wr_off:>13.1f}%                | {wr_on:>13.1f}%")
    print(f" 최고 손실폭(MDD)| {mdd_off:>13.1f}%                | {mdd_on:>13.1f}%")
    print(f" 누적 수익률     | {ret_off:>13.1f}%                | {ret_on:>13.1f}%")
    print(f" 최종 잔고       | {df_off['cap'].iloc[-1]:>14,.0f}원             | {df_on['cap'].iloc[-1]:>14,.0f}원")
    print("=" * 80)
    
    df_off['date'] = pd.to_datetime(df_off['date'])
    df_on['date'] = pd.to_datetime(df_on['date'])
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(df_off['date'], df_off['cap'], color='#FF4500', linewidth=1.5, label=f'Raw GBDT (No Filters) - MDD: {mdd_off:.1f}%')
    ax.plot(df_on['date'], df_on['cap'], color='#00FF00', linewidth=2.5, label=f'Lumos Complete (QQQ 60m EMA) - MDD: {mdd_on:.1f}%')
    
    ax.set_title("Lumos WFA Equity Curve: Shields OFF vs ON (QQQ 60m Mode)", fontsize=18, fontweight='bold', pad=20, color='white')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{y/100000000:,.0f}억' if y >= 100000000 else f'{y/10000:,.0f}만'))
    
    ax.grid(True, linestyle='--', alpha=0.2, color='gray')
    ax.legend(loc='upper left', fontsize=12, facecolor='black', edgecolor='white')
    
    plt.tight_layout()
    chart_path = str(PROJECT_ROOT / 'data' / 'ab_test_wfa.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    run_wfa_ab_test()
