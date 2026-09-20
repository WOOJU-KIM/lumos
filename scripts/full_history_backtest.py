import sys
sys.path.append('.')
import sqlite3
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import time
from core.ml_engine import MLFeatureEngine
from lightgbm import LGBMClassifier
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def generate_synthetic_5m_day(date, day_o, day_h, day_l, day_c):
    times = pd.date_range(start=f"{date.strftime('%Y-%m-%d')} 09:30:00", 
                          end=f"{date.strftime('%Y-%m-%d')} 15:55:00", freq='5min')
    n = len(times)
    if n == 0: return pd.DataFrame()
    t = np.linspace(0, 1, n)
    W = np.random.standard_normal(n).cumsum()
    W = W - t * W[-1]
    path = day_o + t * (day_c - day_o)
    path += W * (day_h - day_l) * 0.1
    path = np.clip(path, day_l, day_h)
    path[0] = day_o
    path[-1] = day_c
    noise = np.random.uniform(-0.001, 0.001, n) * path
    path = path + noise
    df = pd.DataFrame({'datetime': times, 'Close': path})
    df['Open'] = df['Close'].shift(1).fillna(day_o)
    df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.0005, n)))
    df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.0005, n)))
    df['Volume'] = np.random.randint(1000, 50000, n)
    return df

def train_and_predict(w_idx, unique_weeks, tqqq_feat, feature_cols, ROLLING_WINDOW_WEEKS):
    cur_test_week = unique_weeks[w_idx]
    train_weeks = unique_weeks[w_idx - ROLLING_WINDOW_WEEKS : w_idx]
    
    train_mask = tqqq_feat['week_id'].isin(train_weeks)
    X_tr = tqqq_feat.loc[train_mask, feature_cols].fillna(0.0)
    y_tr = tqqq_feat.loc[train_mask, 'target']
    
    if len(X_tr) < 100 or len(y_tr.unique()) < 2:
        return None
        
    clf = LGBMClassifier(
        objective='multiclass', num_class=3, class_weight='balanced',
        n_estimators=80, max_depth=4, learning_rate=0.03,
        random_state=42, verbosity=-1, n_jobs=1
    )
    clf.fit(X_tr, y_tr)
    
    test_mask = tqqq_feat['week_id'] == cur_test_week
    w_test_df = tqqq_feat.loc[test_mask].copy()
    if w_test_df.empty: return None
    
    X_te = w_test_df[feature_cols].fillna(0.0)
    probs = clf.predict_proba(X_te)
    
    w_test_df['Prob_Short'] = probs[:, 0]
    w_test_df['Prob_Neutral'] = probs[:, 1]
    w_test_df['Prob_Long'] = probs[:, 2]
    return w_test_df
    
def get_real_data():
    conn = sqlite3.connect('data/market_data.db')
    try:
        tqqq = pd.read_sql("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='TQQQ'", conn)
        sqqq = pd.read_sql("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='SQQQ'", conn)
        vix = pd.read_sql("SELECT datetime, close FROM market_candles WHERE symbol='^VIX'", conn)
        tnx = pd.read_sql("SELECT datetime, close FROM market_candles WHERE symbol='^TNX'", conn)
        
        tqqq['datetime'] = pd.to_datetime(tqqq['datetime'])
        sqqq['datetime'] = pd.to_datetime(sqqq['datetime'])
        vix['datetime'] = pd.to_datetime(vix['datetime'])
        tnx['datetime'] = pd.to_datetime(tnx['datetime'])
        
        vix = vix.rename(columns={'close': 'VIX'})
        tnx = tnx.rename(columns={'close': 'TNX'})
        
        real_df = pd.merge(tqqq, sqqq[['datetime', 'Close']], on='datetime', how='left', suffixes=('', '_sqqq'))
        real_df = pd.merge_asof(real_df.sort_values('datetime'), vix.sort_values('datetime'), on='datetime')
        real_df = pd.merge_asof(real_df.sort_values('datetime'), tnx.sort_values('datetime'), on='datetime')
        return real_df
    except Exception as e:
        print("Real data error:", e)
        return pd.DataFrame()

def run_full_history():
    print("="*50)
    print("Starting 1980-Present Full History Backtest (WFA)")
    print("="*50)
    
    train_start = '1978-01-01'
    test_end = pd.Timestamp.now().strftime('%Y-%m-%d')
    print("Downloading ^NDX, ^VIX, ^TNX from Yahoo Finance...")
    ndx_daily = yf.download('^NDX', start=train_start, end=test_end, progress=False)
    vix_daily = yf.download('^VIX', start=train_start, end=test_end, progress=False)
    tnx_daily = yf.download('^TNX', start=train_start, end=test_end, progress=False)
    
    df_daily = pd.DataFrame(index=ndx_daily.index)
    df_daily['NDX_O'] = ndx_daily['Open']
    df_daily['NDX_H'] = ndx_daily['High']
    df_daily['NDX_L'] = ndx_daily['Low']
    df_daily['NDX_C'] = ndx_daily['Close']
    
    if len(vix_daily) > 0: df_daily['VIX_C'] = vix_daily['Close']
    else: df_daily['VIX_C'] = np.nan
        
    if len(tnx_daily) > 0: df_daily['TNX_C'] = tnx_daily['Close']
    else: df_daily['TNX_C'] = np.nan
        
    df_daily['VIX_C'] = df_daily['VIX_C'].fillna(20.0 + df_daily['NDX_C'].pct_change().rolling(20).std() * 1000)
    df_daily['TNX_C'] = df_daily['TNX_C'].fillna(5.0)
    df_daily.ffill(inplace=True)
    df_daily.dropna(inplace=True)
    
    tqqq_c = [100.0]; tqqq_o, tqqq_h, tqqq_l = [], [], []
    sqqq_c = [100.0]; sqqq_o, sqqq_h, sqqq_l = [], [], []
    
    ndx_c_prev = df_daily['NDX_C'].iloc[0]
    for i in range(len(df_daily)):
        row = df_daily.iloc[i]
        ret_o = (row['NDX_O'] / ndx_c_prev) - 1.0
        ret_h = (row['NDX_H'] / ndx_c_prev) - 1.0
        ret_l = (row['NDX_L'] / ndx_c_prev) - 1.0
        ret_c = (row['NDX_C'] / ndx_c_prev) - 1.0
        c_prev_t = tqqq_c[-1]
        t_o = max(0.1, c_prev_t * (1 + 3 * ret_o))
        t_h = max(0.1, c_prev_t * (1 + 3 * ret_h))
        t_l = max(0.1, c_prev_t * (1 + 3 * ret_l))
        t_c = max(0.1, c_prev_t * (1 + 3 * ret_c))
        t_h = max(t_o, t_c, t_h); t_l = min(t_o, t_c, t_l)
        tqqq_o.append(t_o); tqqq_h.append(t_h); tqqq_l.append(t_l); tqqq_c.append(t_c)
        
        c_prev_s = sqqq_c[-1]
        s_o = max(0.1, c_prev_s * (1 - 3 * ret_o))
        s_h = max(0.1, c_prev_s * (1 - 3 * ret_l))
        s_l = max(0.1, c_prev_s * (1 - 3 * ret_h))
        s_c = max(0.1, c_prev_s * (1 - 3 * ret_c))
        s_h = max(s_o, s_c, s_h); s_l = min(s_o, s_c, s_l)
        sqqq_o.append(s_o); sqqq_h.append(s_h); sqqq_l.append(s_l); sqqq_c.append(s_c)
        ndx_c_prev = row['NDX_C']
        
    df_daily['TQQQ_O'] = tqqq_o; df_daily['TQQQ_H'] = tqqq_h; df_daily['TQQQ_L'] = tqqq_l; df_daily['TQQQ_C'] = tqqq_c[1:]
    df_daily['SQQQ_O'] = sqqq_o; df_daily['SQQQ_H'] = sqqq_h; df_daily['SQQQ_L'] = sqqq_l; df_daily['SQQQ_C'] = sqqq_c[1:]
    
    real_df = get_real_data()
    real_start_dt = pd.to_datetime('2026-12-31')
    if not real_df.empty:
        real_start_dt = real_df['datetime'].min()
        print(f"Loaded Real Data starting from {real_start_dt}")
        
    print("Generating Master 15m Path for ML Feature Extraction...")
    np.random.seed(42)
    dfs_5m_master = []
    
    df_daily_fake = df_daily[df_daily.index < real_start_dt]
    for dt, row in df_daily_fake.iterrows():
        d5 = generate_synthetic_5m_day(dt, row['TQQQ_O'], row['TQQQ_H'], row['TQQQ_L'], row['TQQQ_C'])
        d5['VIX'] = row['VIX_C']
        d5['TNX'] = row['TNX_C']
        dfs_5m_master.append(d5)
        
    fake_master_5m = pd.concat(dfs_5m_master, ignore_index=True)
    
    if not real_df.empty:
        t_real = real_df.copy()
        t_real['VIX'] = t_real['VIX'].ffill().fillna(20.0)
        t_real['TNX'] = t_real['TNX'].ffill().fillna(5.0)
        master_5m = pd.concat([fake_master_5m, t_real[['datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'VIX', 'TNX']]], ignore_index=True)
    else:
        master_5m = fake_master_5m
        
    tqqq_15m = master_5m.set_index('datetime').resample('15Min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum', 'VIX': 'last', 'TNX': 'last'
    }).dropna().reset_index()
    
    for col in ['NVDA_Close', 'QQQ_Close', 'VIXY_Close', 'IEF_Close']:
        tqqq_15m[col] = tqqq_15m['Close']
        
    print("Extracting features (This might take a minute)...")
    ml = MLFeatureEngine(confidence_threshold=0.60)
    tqqq_feat = ml.extract_features(tqqq_15m)
    labels = ml.compute_triple_barrier_labels(tqqq_feat)
    tqqq_feat['target'] = labels.map({1: 2, -1: 0, 0: 1}).fillna(1).astype(int)
    
    tqqq_feat['datetime_dt'] = pd.to_datetime(tqqq_feat['datetime'])
    tqqq_feat['date_str'] = tqqq_feat['datetime_dt'].dt.strftime('%Y-%m-%d')
    tqqq_feat['week_id'] = tqqq_feat['datetime_dt'].dt.isocalendar().year.astype(str) + "_" + tqqq_feat['datetime_dt'].dt.isocalendar().week.astype(str).str.zfill(2)
    
    raw_price_features = {
        'open', 'high', 'low', 'close', 'volume', 'Open', 'High', 'Low', 'Close', 'Volume',
        'datetime', 'datetime_dt', 'date_str', 'time_str', 'week_id', 'target', 'date', 'year',
        'cum_vp', 'vwap', 'EMA_9', 'EMA_21', 'EMA_50', 'EMA_200',
        'BB_Upper', 'BB_Lower', 'KC_Upper', 'KC_Lower', 'ATR_14', 'MACD', 'MACD_Signal', 'MACD_Hist'
    }
    feature_cols = [c for c in tqqq_feat.columns if c not in raw_price_features and pd.api.types.is_numeric_dtype(tqqq_feat[c])]
    
    unique_weeks = sorted(tqqq_feat['week_id'].unique().tolist())
    ROLLING_WINDOW_WEEKS = 104
    
    target_weeks = list(range(ROLLING_WINDOW_WEEKS, len(unique_weeks)))
    print(f"Running WFA on {len(target_weeks)} weeks (Parallel)...")
    
    preds_list = Parallel(n_jobs=-1, verbose=5)(
        delayed(train_and_predict)(w_idx, unique_weeks, tqqq_feat, feature_cols, ROLLING_WINDOW_WEEKS)
        for w_idx in target_weeks
    )
    
    preds = [p for p in preds_list if p is not None]
    if len(preds) == 0:
        print("No predictions generated.")
        return
        
    tqqq_feat_pred = pd.concat(preds, ignore_index=True)
    
    test_start = '1980-01-01'
    tqqq_feat_pred = tqqq_feat_pred[tqqq_feat_pred['date_str'] >= test_start].copy()
    
    tqqq_feat_pred['Sig'] = 0
    tqqq_feat_pred['ema20'] = tqqq_feat_pred['Close'].ewm(span=20, adjust=False).mean()
    tqqq_feat_pred = tqqq_feat_pred.reset_index(drop=True)
    
    long_mask = (tqqq_feat_pred['Prob_Long'] > tqqq_feat_pred['Prob_Short']) & (tqqq_feat_pred['Prob_Long'] > tqqq_feat_pred['Prob_Neutral']) & (tqqq_feat_pred['Prob_Long'] >= 0.60) & (tqqq_feat_pred['Close'] > tqqq_feat_pred['ema20'])
    short_mask = (tqqq_feat_pred['Prob_Short'] > tqqq_feat_pred['Prob_Long']) & (tqqq_feat_pred['Prob_Short'] > tqqq_feat_pred['Prob_Neutral']) & (tqqq_feat_pred['Prob_Short'] >= 0.60) & (tqqq_feat_pred['Close'] < tqqq_feat_pred['ema20'])
    tqqq_feat_pred.loc[long_mask, 'Sig'] = 1
    tqqq_feat_pred.loc[short_mask, 'Sig'] = -1
    
    print(f"Num Long Signals: {(tqqq_feat_pred['Sig'] == 1).sum()}")
    print(f"Num Short Signals: {(tqqq_feat_pred['Sig'] == -1).sum()}")
    
    tqqq_feat_pred['datetime'] = pd.to_datetime(tqqq_feat_pred['datetime'])
    
    n_paths = 5
    print(f"Running {n_paths} 5m simulation paths...")
    
    df_daily_test = df_daily[(df_daily.index >= test_start) & (df_daily.index < real_start_dt)]
    
    all_paths_cap = []
    final_caps = []
    final_mdds = []
    
    for sim in range(n_paths):
        np.random.seed(sim)
        dfs_5m = []
        dfs_sqqq_5m = []
        
        for dt, row in df_daily_test.iterrows():
            d5 = generate_synthetic_5m_day(dt, row['TQQQ_O'], row['TQQQ_H'], row['TQQQ_L'], row['TQQQ_C'])
            s5 = generate_synthetic_5m_day(dt, row['SQQQ_O'], row['SQQQ_H'], row['SQQQ_L'], row['SQQQ_C'])
            dfs_5m.append(d5)
            dfs_sqqq_5m.append(s5)
            
        tqqq_5m = pd.concat(dfs_5m, ignore_index=True)
        sqqq_5m = pd.concat(dfs_sqqq_5m, ignore_index=True)
        tqqq_5m['SQQQ_Close'] = sqqq_5m['Close']
        
        if not real_df.empty:
            r5 = real_df[['datetime', 'Open', 'High', 'Low', 'Close', 'Close_sqqq']].copy()
            r5 = r5.rename(columns={'Close_sqqq': 'SQQQ_Close'})
            tqqq_5m = pd.concat([tqqq_5m, r5], ignore_index=True)
            
        tqqq_5m['datetime'] = pd.to_datetime(tqqq_5m['datetime'])
        
        df_merged = pd.merge_asof(tqqq_5m.sort_values('datetime'), 
                                  tqqq_feat_pred[['datetime', 'Sig']].sort_values('datetime'),
                                  on='datetime', direction='backward')
                                  
        capital = 10_000_000.0
        pos = 0
        entry_px = 0
        peak_px = 0
        cap_curve = []
        dates = []
        current_sym = None
        
        for row in df_merged.itertuples():
            sig = row.Sig
            
            if pos == 0:
                if sig == 1:
                    current_sym = 'TQQQ'
                    cur_px = row.Close
                    pos = (capital * 0.98) / cur_px
                    entry_px = cur_px
                    peak_px = entry_px
                    capital -= pos * entry_px
                elif sig == -1:
                    current_sym = 'SQQQ'
                    cur_px = row.SQQQ_Close
                    if pd.isna(cur_px): continue
                    pos = (capital * 0.98) / cur_px
                    entry_px = cur_px
                    peak_px = entry_px
                    capital -= pos * entry_px
            elif pos > 0:
                if current_sym == 'TQQQ':
                    cur_px = row.Close
                else:
                    cur_px = row.SQQQ_Close
                    
                if pd.isna(cur_px): continue
                if cur_px > peak_px: peak_px = cur_px
                
                ret = (cur_px / entry_px) - 1.0
                max_ret = (peak_px / entry_px) - 1.0
                
                sell = False
                if max_ret >= 0.015 and cur_px <= peak_px * (1 - 0.003): sell = True
                elif ret <= -0.02: sell = True
                elif ret >= 0.10: sell = True
                
                if sell:
                    capital += pos * cur_px * (1 - 0.0020)
                    pos = 0
                    current_sym = None
                    cap_curve.append(capital)
                    dates.append(row.datetime)
                    
        if pos > 0:
            if current_sym == 'TQQQ':
                last_px = df_merged.iloc[-1]['Close']
            else:
                last_px = df_merged.iloc[-1]['SQQQ_Close']
            capital += pos * last_px * (1 - 0.0020)
            cap_curve.append(capital)
            dates.append(df_merged.iloc[-1]['datetime'])
            
        mdd = 0
        if len(cap_curve) > 0:
            cc = np.array(cap_curve)
            rm = np.maximum.accumulate(cc)
            mdd = np.max((rm - cc) / rm) * 100.0
            
        print(f"Path {sim+1}: Final Cap={capital:,.0f}, MDD={mdd:.2f}%")
        
        path_df = pd.DataFrame({'datetime': dates, 'capital': cap_curve})
        path_df = path_df.set_index('datetime')
        path_df = path_df[~path_df.index.duplicated(keep='last')]
        all_paths_cap.append(path_df['capital'])
        final_caps.append(capital)
        final_mdds.append(mdd)

    plt.figure(figsize=(14, 7))
    for i, c in enumerate(all_paths_cap):
        plt.plot(c, label=f'Path {i+1}', alpha=0.7)
        
    plt.yscale('log')
    plt.title('1980-2026 Full History WFA Backtest (Log Scale)')
    plt.xlabel('Date')
    plt.ylabel('Capital (KRW)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/full_history_chart.png')
    print("Chart saved to data/full_history_chart.png")

if __name__ == '__main__':
    run_full_history()
