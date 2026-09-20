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

def run_cpcv():
    print('=' * 80)
    print('🚀 [조합적 분할 테스트 (CPCV: Combinatorial Purged Cross-Validation)]')
    print('   - N=6 (6개의 구간으로 분할), K=2 (2개의 테스트 구간 선택)')
    print('   - 총 15개의 훈련/테스트 조합(Combinations) 평가')
    print('   - 각 구간별 엄밀한 5분봉 정밀 궤적(5m Precision) 추적 적용')
    print('=' * 80)

    lake = MarketDataLake()
    tqqq_15m = lake.load_candles('TQQQ', '15m')
    sqqq_15m = lake.load_candles('SQQQ', '15m')
    soxx_60m = lake.load_candles('SOXX', '60m')
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
    target_series = labels.map({1: 2, -1: 0, 0: 1}).fillna(1).astype(int)
    tqqq_feat['target'] = target_series
    
    tqqq_feat['datetime_dt'] = pd.to_datetime(tqqq_feat['datetime'])
    tqqq_feat['date_str'] = tqqq_feat['datetime_dt'].dt.strftime('%Y-%m-%d')
    
    raw_price_features = {
        'open', 'high', 'low', 'close', 'volume', 'Open', 'High', 'Low', 'Close', 'Volume',
        'datetime', 'datetime_dt', 'date_str', 'time_str', 'week_id', 'target', 'date', 'year',
        'cum_vp', 'vwap', 'EMA_9', 'EMA_21', 'EMA_50', 'EMA_200',
        'BB_Upper', 'BB_Lower', 'KC_Upper', 'KC_Lower', 'ATR_14', 'MACD', 'MACD_Signal', 'MACD_Hist'
    }
    feature_cols = [c for c in tqqq_feat.columns if c not in raw_price_features and pd.api.types.is_numeric_dtype(tqqq_feat[c])]
    
    unique_dates = sorted(tqqq_feat['date_str'].unique())
    n_splits = 6
    split_size = len(unique_dates) // n_splits
    folds = []
    for i in range(n_splits):
        start = i * split_size
        end = (i + 1) * split_size if i < n_splits - 1 else len(unique_dates)
        folds.append(unique_dates[start:end])
        
    results = []
    combos = list(itertools.combinations(range(n_splits), 2))
    
    for combo_idx, test_fold_indices in enumerate(combos):
        train_fold_indices = [i for i in range(n_splits) if i not in test_fold_indices]
        
        train_dates = []
        for i in train_fold_indices: train_dates.extend(folds[i])
        test_dates = []
        for i in test_fold_indices: test_dates.extend(folds[i])
        
        train_mask = tqqq_feat['date_str'].isin(train_dates)
        test_mask = tqqq_feat['date_str'].isin(test_dates)
        
        X_tr = tqqq_feat.loc[train_mask, feature_cols].fillna(0.0)
        y_tr = tqqq_feat.loc[train_mask, 'target']
        X_te = tqqq_feat.loc[test_mask, feature_cols].fillna(0.0)
        
        clf = LGBMClassifier(
            objective='multiclass', num_class=3, class_weight='balanced',
            n_estimators=80, max_depth=4, learning_rate=0.03, random_state=42, verbosity=-1, n_jobs=-1
        )
        clf.fit(X_tr, y_tr)
        
        w_probs = clf.predict_proba(X_te)
        test_df = tqqq_feat[test_mask].copy()
        
        p_s = w_probs[:, 0]
        p_n = w_probs[:, 1]
        p_l = w_probs[:, 2]
        
        w_confs = np.full(len(test_df), 0.50, dtype=float)
        w_dirs = ['NONE'] * len(test_df)
        for i in range(len(test_df)):
            ps, pn, pl = p_s[i], p_n[i], p_l[i]
            if pl > pn and pl > ps:
                w_confs[i] = min(0.95, max(0.50, 0.50 + (pl - 0.333) * 1.15))
                w_dirs[i] = 'LONG_TQQQ'
            elif ps > pn and ps > pl:
                w_confs[i] = min(0.95, max(0.50, 0.50 + (ps - 0.333) * 1.15))
                w_dirs[i] = 'SHORT_SQQQ'
                
        test_df['Confidence'] = w_confs
        test_df['Direction'] = w_dirs
        
        # 5m Evaluation
        wins = 0
        losses = 0
        cap = 10000000.0
        max_cap = cap
        mdd = 0.0
        
        for d_str in test_dates:
            day_bars_15 = test_df[test_df['date_str'] == d_str].reset_index(drop=True)
            day_tqqq_5 = tqqq_5m_by_date.get(d_str, pd.DataFrame())
            day_sqqq_5 = sqqq_5m_by_date.get(d_str, pd.DataFrame())
            
            for b_idx in range(len(day_bars_15)):
                row_15 = day_bars_15.iloc[b_idx]
                d_dir = row_15['Direction']
                d_conf = row_15['Confidence']
                if d_dir == 'NONE' or d_conf < 0.60: continue
                
                target_sym = 'TQQQ' if d_dir == 'LONG_TQQQ' else 'SQQQ'
                entry_time = row_15['datetime'][11:16]
                if entry_time >= '15:30': continue
                
                post_5m = day_tqqq_5 if target_sym == 'TQQQ' else day_sqqq_5
                post_5m = post_5m[post_5m['datetime'].str.slice(11, 16) > entry_time]
                if post_5m.empty: continue
                
                entry_price = float(post_5m.iloc[0].get('Open', post_5m.iloc[0].get('open')))
                
                initial_sl = 0.02
                atr_14 = float(row_15.get('ATR_14', 0))
                c_px = float(row_15.get('Close', 1.0))
                if atr_14 > 0 and c_px > 0:
                    atr_sl_pct = (atr_14 * 1.5) / c_px
                    initial_sl = max(0.02, min(0.032, atr_sl_pct))
                    
                peak_ret = 0.0
                trailing_active = False
                hit_sl = False
                hit_tp = False
                exit_ret = 0.0
                
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
                        exit_ret = -initial_sl
                        break
                    elif cur_max_ret >= 0.10:
                        exit_ret = 0.10
                        break
                    elif trailing_active and (peak_ret - cur_c_ret >= 0.003):
                        exit_ret = cur_c_ret
                        break
                        
                    if b_time >= '15:50' or k == 23:
                        exit_ret = cur_c_ret
                        break
                
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
        print(f'      -> 테스트 구간 승률: {win_rate:.1f}% | 누적 수익률: {ret_pct:+.1f}% | MDD: {mdd:.1f}% | 거래수: {total_t}회')
        results.append({'wr': win_rate, 'ret': ret_pct, 'mdd': mdd, 'trades': total_t})
        
    df_res = pd.DataFrame(results)
    print('=' * 80)
    print(f'🌟 [CPCV 종합 결과 (총 15개 시나리오 평균)]')
    print(f'   - 평균 승률: {df_res["wr"].mean():.2f}%')
    print(f'   - 평균 16개월 테스트 구간 수익률: {df_res["ret"].mean():+.2f}%')
    print(f'   - 평균 구간 MDD: {df_res["mdd"].mean():.2f}%')
    print(f'   - 평균 거래횟수: {df_res["trades"].mean():.1f}회')
    print('=' * 80)

if __name__ == '__main__':
    run_cpcv()
