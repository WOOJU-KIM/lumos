import sys
import numpy as np
import pandas as pd
from pathlib import Path
import itertools
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))
from core.data_lake import MarketDataLake
from core.ml_engine import MLFeatureEngine
from core.heterogeneous_models import CrossAssetDislocationModel

def run_cpcv_final():
    print('=' * 80)
    print('🚀 [CPCV 방어막 완전체 (Cross-Asset Veto + 60m SOXX 추세 + 보수적 5m)]')
    print('   - N=6, K=2 (15가지 시나리오 극한 테스트)')
    print('=' * 80)

    lake = MarketDataLake()
    tqqq_15m = lake.load_candles('TQQQ', '15m')
    sqqq_15m = lake.load_candles('SQQQ', '15m')
    soxx_60m = lake.load_candles('SOXX', '60m')
    nvda_15m = lake.load_candles('NVDA', '15m')
    soxx_15m = lake.load_candles('SOXX', '15m')
    qqq_15m  = lake.load_candles('QQQ', '15m')
    vixy_15m = lake.load_candles('VIXY', '15m')
    ief_15m  = lake.load_candles('IEF', '15m')
    tqqq_5m = lake.load_candles('TQQQ', '5m')
    sqqq_5m = lake.load_candles('SQQQ', '5m')
    
    soxx_60m['ema20'] = soxx_60m['Close'].ewm(span=20, adjust=False).mean()
    
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
    
    tqqq_feat['ATR_Pct'] = (tqqq_feat['ATR_14'] / (tqqq_feat['Close'] + 1e-9)) * 100.0
    tqqq_feat['MACD_Pct'] = (tqqq_feat['MACD'] / (tqqq_feat['Close'] + 1e-9)) * 100.0
    
    raw_price_features = {'open', 'high', 'low', 'close', 'volume', 'Open', 'High', 'Low', 'Close', 'Volume',
        'datetime', 'datetime_dt', 'date_str', 'time_str', 'week_id', 'target', 'date', 'year',
        'cum_vp', 'vwap', 'EMA_9', 'EMA_21', 'EMA_50', 'EMA_200',
        'BB_Upper', 'BB_Lower', 'KC_Upper', 'KC_Lower', 'ATR_14', 'MACD', 'MACD_Signal', 'MACD_Hist'}
    feature_cols = [c for c in tqqq_feat.columns if c not in raw_price_features and pd.api.types.is_numeric_dtype(tqqq_feat[c])]
    
    nvda_map = nvda_15m.set_index('datetime')['Close'].to_dict()
    soxx_map = soxx_15m.set_index('datetime')['Close'].to_dict()
    qqq_map  = qqq_15m.set_index('datetime')['Close'].to_dict()
    vix_map  = vixy_15m.set_index('datetime')['Close'].to_dict()
    ief_map  = ief_15m.set_index('datetime')['Close'].to_dict()
    
    cross_mod = CrossAssetDislocationModel(dislocation_z_threshold=1.6)
    
    unique_dates = sorted(tqqq_feat['date_str'].unique())
    n_splits = 6
    split_size = len(unique_dates) // n_splits
    folds = [unique_dates[i*split_size : (i+1)*split_size if i<n_splits-1 else len(unique_dates)] for i in range(n_splits)]
        
    results = []
    combos = list(itertools.combinations(range(n_splits), 2))
    
    for combo_idx, test_fold_indices in enumerate(combos):
        train_fold_indices = [i for i in range(n_splits) if i not in test_fold_indices]
        train_dates = [d for i in train_fold_indices for d in folds[i]]
        test_dates = [d for i in test_fold_indices for d in folds[i]]
        
        train_mask = tqqq_feat['date_str'].isin(train_dates)
        test_mask = tqqq_feat['date_str'].isin(test_dates)
        
        clf = LGBMClassifier(
            objective='multiclass', num_class=3, class_weight='balanced',
            n_estimators=80, max_depth=4, learning_rate=0.03, random_state=42, verbosity=-1, n_jobs=-1
        )
        clf.fit(tqqq_feat.loc[train_mask, feature_cols].fillna(0.0), tqqq_feat.loc[train_mask, 'target'])
        
        w_probs = clf.predict_proba(tqqq_feat.loc[test_mask, feature_cols].fillna(0.0))
        test_df = tqqq_feat[test_mask].copy()
        
        w_confs = np.full(len(test_df), 0.50, dtype=float)
        w_dirs = ['NONE'] * len(test_df)
        for i in range(len(test_df)):
            ps, pn, pl = w_probs[i, 0], w_probs[i, 1], w_probs[i, 2]
            if pl > pn and pl > ps:
                w_confs[i] = min(0.95, max(0.50, 0.50 + (pl - 0.333) * 1.15))
                w_dirs[i] = 'LONG_TQQQ'
            elif ps > pn and ps > pl:
                w_confs[i] = min(0.95, max(0.50, 0.50 + (ps - 0.333) * 1.15))
                w_dirs[i] = 'SHORT_SQQQ'
                
        test_df['Confidence'] = w_confs
        test_df['Direction'] = w_dirs
        
        wins = 0; losses = 0; cap = 10000000.0; max_cap = cap; mdd = 0.0
        
        for d_str in test_dates:
            day_bars_15 = test_df[test_df['date_str'] == d_str].reset_index(drop=True)
            day_tqqq_5 = tqqq_5m_by_date.get(d_str, pd.DataFrame())
            day_sqqq_5 = sqqq_5m_by_date.get(d_str, pd.DataFrame())
            
            for b_idx in range(len(day_bars_15)):
                if b_idx < 5: continue
                row = day_bars_15.iloc[b_idx]
                dt_str = row['datetime']
                entry_time = dt_str[11:16]
                if entry_time >= '15:30': continue
                
                dir_gbdt = row['Direction']
                conf_gbdt = row['Confidence']
                if dir_gbdt == 'NONE' or conf_gbdt < 0.60: continue
                
                cur_tqqq_close = float(row.get('Close', 1.0))
                cur_sqqq_close = 1.0
                
                d_date = dt_str[:10]
                soxx_day = soxx_60m[soxx_60m['datetime'].str.startswith(d_date)]
                soxx_60m_bull = True
                soxx_60m_bear = True
                if not soxx_day.empty:
                    last_soxx = soxx_day.iloc[-1]
                    s_close = float(last_soxx['Close'])
                    s_ema = float(last_soxx['ema20'])
                    if s_ema > 0:
                        if s_close < s_ema: soxx_60m_bull = False
                        if s_close > s_ema: soxx_60m_bear = False
                
                # Cross Asset Veto
                cross_dir = "NONE"
                n_px = nvda_map.get(dt_str)
                sx_px = soxx_map.get(dt_str)
                q_px = qqq_map.get(dt_str)
                v_px = vix_map.get(dt_str)
                i_px = ief_map.get(dt_str)
                
                if n_px and sx_px and q_px:
                    prev_dt = day_bars_15.iloc[b_idx - 5]['datetime']
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
                    tnx_proxy_ret = -ief_r
                    
                    tqqq_r = (cur_tqqq_close / day_bars_15.iloc[b_idx - 5]['Close']) - 1.0
                    sig_code, _, _ = cross_mod.predict_signal(
                        tqqq_ret=tqqq_r, nvda_ret=nvda_r, soxx_ret=soxx_r,
                        qqq_ret=qqq_r, vix_ret=vix_r, tnx_ret=tnx_proxy_ret
                    )
                    if sig_code > 0: cross_dir = "LONG_TQQQ"
                    elif sig_code < 0: cross_dir = "SHORT_SQQQ"
                
                entry_approved = False
                target_sym = None
                
                if dir_gbdt == "LONG_TQQQ" and soxx_60m_bull and cross_dir != "SHORT_SQQQ":
                    entry_approved = True; target_sym = "TQQQ"
                elif dir_gbdt == "SHORT_SQQQ" and soxx_60m_bear and cross_dir != "LONG_TQQQ":
                    entry_approved = True; target_sym = "SQQQ"
                    
                if not entry_approved: continue
                
                post_5m = day_tqqq_5 if target_sym == 'TQQQ' else day_sqqq_5
                post_5m = post_5m[post_5m['datetime'].str.slice(11, 16) > entry_time]
                if post_5m.empty: continue
                
                entry_price = float(post_5m.iloc[0].get('Open', post_5m.iloc[0].get('open')))
                atr_14 = float(row.get('ATR_14', 0))
                c_px = float(row.get('Close', 1.0))
                initial_sl = 0.02
                if atr_14 > 0 and c_px > 0:
                    initial_sl = max(0.02, min(0.032, (atr_14 * 1.5) / c_px))
                    
                peak_ret = 0.0; trailing_active = False; exit_ret = 0.0
                
                for k in range(len(post_5m)):
                    b_5m = post_5m.iloc[k]
                    b_h = float(b_5m.get('High', b_5m.get('high')))
                    b_l = float(b_5m.get('Low', b_5m.get('low')))
                    b_c = float(b_5m.get('Close', b_5m.get('close')))
                    b_time = b_5m['datetime'][11:16]
                    
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
                
        total_t = wins + losses
        win_rate = (wins / total_t * 100) if total_t > 0 else 0
        ret_pct = (cap - 10000000.0) / 10000000.0 * 100.0
        
        print(f'   [조합 {combo_idx+1:02d}/15] Train Fold: {train_fold_indices} | Test Fold: {test_fold_indices}')
        print(f'      -> 승률: {win_rate:.1f}% | 누적 수익률: {ret_pct:+.1f}% | MDD: {mdd:.1f}% | 거래수: {total_t}회')
        results.append({'wr': win_rate, 'ret': ret_pct, 'mdd': mdd, 'trades': total_t})
        
    df_res = pd.DataFrame(results)
    print('=' * 80)
    print(f'🌟 [CPCV 완전체 종합 결과]')
    print(f'   - 평균 승률: {df_res["wr"].mean():.2f}%')
    print(f'   - 평균 누적 수익률: {df_res["ret"].mean():+.2f}%')
    print(f'   - 평균 구간 MDD: {df_res["mdd"].mean():.2f}%')
    print(f'   - 평균 거래횟수: {df_res["trades"].mean():.1f}회')
    print('=' * 80)

if __name__ == '__main__':
    run_cpcv_final()
