import sys
sys.path.append('.')
import sqlite3
import yfinance as yf
import config
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

def train_and_predict(w_idx, unique_weeks, long_feat, feature_cols, ROLLING_WINDOW_WEEKS):
    cur_test_week = unique_weeks[w_idx]
    train_weeks = unique_weeks[w_idx - ROLLING_WINDOW_WEEKS : w_idx]
    
    train_mask = long_feat['week_id'].isin(train_weeks)
    X_tr = long_feat.loc[train_mask, feature_cols].fillna(0.0)
    y_tr = long_feat.loc[train_mask, 'target']
    
    if len(X_tr) < 100 or len(y_tr.unique()) < 2:
        return None
        
    clf = LGBMClassifier(
        objective='multiclass', num_class=3, class_weight='balanced',
        n_estimators=80, max_depth=4, learning_rate=0.03,
        random_state=42, verbosity=-1, n_jobs=1
    )
    clf.fit(X_tr, y_tr)
    
    test_mask = long_feat['week_id'] == cur_test_week
    w_test_df = long_feat.loc[test_mask].copy()
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
        tqqq = pd.read_sql("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol=config.BASE_ASSET_LONG", conn)
        sqqq = pd.read_sql("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol=config.BASE_ASSET_SHORT", conn)
        vix = pd.read_sql("SELECT datetime, close FROM market_candles WHERE symbol='^VIX'", conn)
        tnx = pd.read_sql("SELECT datetime, close FROM market_candles WHERE symbol='^TNX'", conn)
        
        tqqq['datetime'] = pd.to_datetime(tqqq['datetime'])
        sqqq['datetime'] = pd.to_datetime(sqqq['datetime'])
        vix['datetime'] = pd.to_datetime(vix['datetime'])
        tnx['datetime'] = pd.to_datetime(tnx['datetime'])
        
        vix = vix.rename(columns={'close': 'VIX'})
        tnx = tnx.rename(columns={'close': 'TNX'})
        
        real_df = pd.merge(tqqq, sqqq[['datetime', 'Close']], on='datetime', how='left', suffixes=('', '_short'))
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
    
    long_c = [100.0]; long_o, long_h, long_l = [], [], []
    short_c = [100.0]; short_o, short_h, short_l = [], [], []
    
    ndx_c_prev = df_daily['NDX_C'].iloc[0]
    for i in range(len(df_daily)):
        row = df_daily.iloc[i]
        ret_o = (row['NDX_O'] / ndx_c_prev) - 1.0
        ret_h = (row['NDX_H'] / ndx_c_prev) - 1.0
        ret_l = (row['NDX_L'] / ndx_c_prev) - 1.0
        ret_c = (row['NDX_C'] / ndx_c_prev) - 1.0
        c_prev_t = long_c[-1]
        t_o = max(0.1, c_prev_t * (1 + 3 * ret_o))
        t_h = max(0.1, c_prev_t * (1 + 3 * ret_h))
        t_l = max(0.1, c_prev_t * (1 + 3 * ret_l))
        t_c = max(0.1, c_prev_t * (1 + 3 * ret_c))
        t_h = max(t_o, t_c, t_h); t_l = min(t_o, t_c, t_l)
        long_o.append(t_o); long_h.append(t_h); long_l.append(t_l); long_c.append(t_c)
        
        c_prev_s = short_c[-1]
        s_o = max(0.1, c_prev_s * (1 - 3 * ret_o))
        s_h = max(0.1, c_prev_s * (1 - 3 * ret_l))
        s_l = max(0.1, c_prev_s * (1 - 3 * ret_h))
        s_c = max(0.1, c_prev_s * (1 - 3 * ret_c))
        s_h = max(s_o, s_c, s_h); s_l = min(s_o, s_c, s_l)
        short_o.append(s_o); short_h.append(s_h); short_l.append(s_l); short_c.append(s_c)
        ndx_c_prev = row['NDX_C']
        
    df_daily['TQQQ_O'] = long_o; df_daily['TQQQ_H'] = long_h; df_daily['TQQQ_L'] = long_l; df_daily['TQQQ_C'] = long_c[1:]
    df_daily['SQQQ_O'] = short_o; df_daily['SQQQ_H'] = short_h; df_daily['SQQQ_L'] = short_l; df_daily['SQQQ_C'] = short_c[1:]
    
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
        
    long_15m = master_5m.set_index('datetime').resample('15Min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum', 'VIX': 'last', 'TNX': 'last'
    }).dropna().reset_index()
    
    for col in ['NVDA_Close', 'QQQ_Close', 'VIXY_Close', 'IEF_Close']:
        long_15m[col] = long_15m['Close']
        
    print("Extracting features (This might take a minute)...")
    ml = MLFeatureEngine(confidence_threshold=0.60)
    long_feat = ml.extract_features(long_15m)
    labels = ml.compute_triple_barrier_labels(long_feat)
    long_feat['target'] = labels.map({1: 2, -1: 0, 0: 1}).fillna(1).astype(int)
    
    long_feat['datetime_dt'] = pd.to_datetime(long_feat['datetime'])
    long_feat['date_str'] = long_feat['datetime_dt'].dt.strftime('%Y-%m-%d')
    long_feat['week_id'] = long_feat['datetime_dt'].dt.isocalendar().year.astype(str) + "_" + long_feat['datetime_dt'].dt.isocalendar().week.astype(str).str.zfill(2)
    
    raw_price_features = {
        'open', 'high', 'low', 'close', 'volume', 'Open', 'High', 'Low', 'Close', 'Volume',
        'datetime', 'datetime_dt', 'date_str', 'time_str', 'week_id', 'target', 'date', 'year',
        'cum_vp', 'vwap', 'EMA_9', 'EMA_21', 'EMA_50', 'EMA_200',
        'BB_Upper', 'BB_Lower', 'KC_Upper', 'KC_Lower', 'ATR_14', 'MACD', 'MACD_Signal', 'MACD_Hist'
    }
    feature_cols = [c for c in long_feat.columns if c not in raw_price_features and pd.api.types.is_numeric_dtype(long_feat[c])]
    
    unique_weeks = sorted(long_feat['week_id'].unique().tolist())
    ROLLING_WINDOW_WEEKS = 104
    
    target_weeks = list(range(ROLLING_WINDOW_WEEKS, len(unique_weeks)))
    print(f"Running WFA on {len(target_weeks)} weeks (Parallel)...")
    
    preds_list = Parallel(n_jobs=-1, verbose=5)(
        delayed(train_and_predict)(w_idx, unique_weeks, long_feat, feature_cols, ROLLING_WINDOW_WEEKS)
        for w_idx in target_weeks
    )
    
    preds = [p for p in preds_list if p is not None]
    if len(preds) == 0:
        print("No predictions generated.")
        return
        
    long_feat_pred = pd.concat(preds, ignore_index=True)
    
    test_start = '1980-01-01'
    long_feat_pred = long_feat_pred[long_feat_pred['date_str'] >= test_start].copy()
    
    long_feat_pred['Sig'] = 0
    long_feat_pred['ema20'] = long_feat_pred['Close'].ewm(span=20, adjust=False).mean()
    long_feat_pred = long_feat_pred.reset_index(drop=True)
    
    long_mask = (long_feat_pred['Prob_Long'] > long_feat_pred['Prob_Short']) & (long_feat_pred['Prob_Long'] > long_feat_pred['Prob_Neutral']) & (long_feat_pred['Prob_Long'] >= 0.60) & (long_feat_pred['Close'] > long_feat_pred['ema20'])
    short_mask = (long_feat_pred['Prob_Short'] > long_feat_pred['Prob_Long']) & (long_feat_pred['Prob_Short'] > long_feat_pred['Prob_Neutral']) & (long_feat_pred['Prob_Short'] >= 0.60) & (long_feat_pred['Close'] < long_feat_pred['ema20'])
    long_feat_pred.loc[long_mask, 'Sig'] = 1
    long_feat_pred.loc[short_mask, 'Sig'] = -1
    
    print(f"Num Long Signals: {(long_feat_pred['Sig'] == 1).sum()}")
    print(f"Num Short Signals: {(long_feat_pred['Sig'] == -1).sum()}")
    
    long_feat_pred['datetime'] = pd.to_datetime(long_feat_pred['datetime'])
    
    n_paths = 5
    print(f"Running {n_paths} 5m simulation paths...")
    
    df_daily_test = df_daily[(df_daily.index >= test_start) & (df_daily.index < real_start_dt)]
    
    all_paths_cap = []
    final_caps = []
    final_mdds = []
    
    for sim in range(n_paths):
        np.random.seed(sim)
        dfs_5m = []
        dfs_short_5m = []
        
        for dt, row in df_daily_test.iterrows():
            d5 = generate_synthetic_5m_day(dt, row['TQQQ_O'], row['TQQQ_H'], row['TQQQ_L'], row['TQQQ_C'])
            s5 = generate_synthetic_5m_day(dt, row['SQQQ_O'], row['SQQQ_H'], row['SQQQ_L'], row['SQQQ_C'])
            dfs_5m.append(d5)
            dfs_short_5m.append(s5)
            
        long_5m = pd.concat(dfs_5m, ignore_index=True)
        short_5m = pd.concat(dfs_short_5m, ignore_index=True)
        long_5m['SQQQ_Close'] = short_5m['Close']
        
        if not real_df.empty:
            r5 = real_df[['datetime', 'Open', 'High', 'Low', 'Close', 'Close_short']].copy()
            r5 = r5.rename(columns={'Close_short': 'SQQQ_Close'})
            long_5m = pd.concat([long_5m, r5], ignore_index=True)
            
        long_5m['datetime'] = pd.to_datetime(long_5m['datetime'])
        
        df_merged = pd.merge_asof(long_5m.sort_values('datetime'), 
                                  long_feat_pred[['datetime', 'Sig']].sort_values('datetime'),
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
                    current_sym = config.BASE_ASSET_LONG
                    cur_px = row.Close
                    pos = (capital * 0.98) / cur_px
                    entry_px = cur_px
                    peak_px = entry_px
                    capital -= pos * entry_px
                elif sig == -1:
                    current_sym = config.BASE_ASSET_SHORT
                    cur_px = row.SQQQ_Close
                    if pd.isna(cur_px): continue
                    pos = (capital * 0.98) / cur_px
                    entry_px = cur_px
                    peak_px = entry_px
                    capital -= pos * entry_px
            elif pos > 0:
                if current_sym == config.BASE_ASSET_LONG:
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
            if current_sym == config.BASE_ASSET_LONG:
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
