import sys
sys.path.append('.')
import sqlite3
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from core.ml_engine import MLFeatureEngine
from lightgbm import LGBMClassifier
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

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

def run_real_backtest():
    print("Loading 100% Real Data from Database...")
    conn = sqlite3.connect('data/market_data.db')
    
    tqqq = pd.read_sql("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='TQQQ' ORDER BY datetime", conn)
    sqqq = pd.read_sql("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='SQQQ' ORDER BY datetime", conn)
    vix = pd.read_sql("SELECT datetime, close as VIX FROM market_candles WHERE symbol='^VIX' ORDER BY datetime", conn)
    tnx = pd.read_sql("SELECT datetime, close as TNX FROM market_candles WHERE symbol='^TNX' ORDER BY datetime", conn)
    nvda = pd.read_sql("SELECT datetime, close as NVDA_Close FROM market_candles WHERE symbol='NVDA' ORDER BY datetime", conn)
    qqq = pd.read_sql("SELECT datetime, close as QQQ_Close FROM market_candles WHERE symbol='QQQ' ORDER BY datetime", conn)
    vixy = pd.read_sql("SELECT datetime, close as VIXY_Close FROM market_candles WHERE symbol='VIXY' ORDER BY datetime", conn)
    ief = pd.read_sql("SELECT datetime, close as IEF_Close FROM market_candles WHERE symbol='IEF' ORDER BY datetime", conn)
    
    tqqq['datetime'] = pd.to_datetime(tqqq['datetime'])
    sqqq['datetime'] = pd.to_datetime(sqqq['datetime'])
    vix['datetime'] = pd.to_datetime(vix['datetime'])
    tnx['datetime'] = pd.to_datetime(tnx['datetime'])
    nvda['datetime'] = pd.to_datetime(nvda['datetime'])
    qqq['datetime'] = pd.to_datetime(qqq['datetime'])
    vixy['datetime'] = pd.to_datetime(vixy['datetime'])
    ief['datetime'] = pd.to_datetime(ief['datetime'])
    
    master_5m = tqqq.copy()
    master_5m = pd.merge(master_5m, sqqq[['datetime', 'Close']], on='datetime', how='left', suffixes=('', '_sqqq'))
    master_5m = master_5m.rename(columns={'Close_sqqq': 'SQQQ_Close'})
    
    master_5m = pd.merge_asof(master_5m, vix, on='datetime')
    master_5m = pd.merge_asof(master_5m, tnx, on='datetime')
    master_5m = pd.merge_asof(master_5m, nvda, on='datetime')
    master_5m = pd.merge_asof(master_5m, qqq, on='datetime')
    master_5m = pd.merge_asof(master_5m, vixy, on='datetime')
    master_5m = pd.merge_asof(master_5m, ief, on='datetime')
    
    master_5m['VIX'] = master_5m['VIX'].fillna(20.0)
    master_5m['TNX'] = master_5m['TNX'].fillna(5.0)
    
    master_5m.ffill(inplace=True)
    master_5m.bfill(inplace=True) 
    master_5m = master_5m.reset_index(drop=True)
    
    print(f"Data Date Range: {master_5m['datetime'].min()} ~ {master_5m['datetime'].max()}")
    
    print("Resampling to 15m and Extracting Features...")
    df_15m = master_5m.set_index('datetime').resample('15Min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum',
        'VIX': 'last', 'TNX': 'last', 'NVDA_Close': 'last', 'QQQ_Close': 'last', 
        'VIXY_Close': 'last', 'IEF_Close': 'last'
    }).dropna().reset_index()
    
    ml = MLFeatureEngine(confidence_threshold=0.60)
    tqqq_feat = ml.extract_features(df_15m)
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
    
    if len(target_weeks) == 0:
        print("Not enough data to run a 104-week WFA! Real data is too short.")
        return
        
    print(f"Running 2-Year Rolling WFA on {len(target_weeks)} weeks (Parallel)...")
    preds_list = Parallel(n_jobs=-1, verbose=0)(
        delayed(train_and_predict)(w_idx, unique_weeks, tqqq_feat, feature_cols, ROLLING_WINDOW_WEEKS)
        for w_idx in target_weeks
    )
    
    preds = [p for p in preds_list if p is not None]
    if len(preds) == 0:
        print("No predictions generated.")
        return
        
    tqqq_feat_pred = pd.concat(preds, ignore_index=True)
    tqqq_feat_pred['Sig'] = 0
    tqqq_feat_pred['ema20'] = tqqq_feat_pred['Close'].ewm(span=20, adjust=False).mean()
    tqqq_feat_pred = tqqq_feat_pred.reset_index(drop=True)
    
    long_mask = (tqqq_feat_pred['Prob_Long'] > tqqq_feat_pred['Prob_Short']) & (tqqq_feat_pred['Prob_Long'] > tqqq_feat_pred['Prob_Neutral']) & (tqqq_feat_pred['Prob_Long'] >= 0.60) & (tqqq_feat_pred['Close'] > tqqq_feat_pred['ema20'])
    short_mask = (tqqq_feat_pred['Prob_Short'] > tqqq_feat_pred['Prob_Long']) & (tqqq_feat_pred['Prob_Short'] > tqqq_feat_pred['Prob_Neutral']) & (tqqq_feat_pred['Prob_Short'] >= 0.60) & (tqqq_feat_pred['Close'] < tqqq_feat_pred['ema20'])
    tqqq_feat_pred.loc[long_mask, 'Sig'] = 1
    tqqq_feat_pred.loc[short_mask, 'Sig'] = -1
    
    print(f"Num Long Signals: {(tqqq_feat_pred['Sig'] == 1).sum()}")
    print(f"Num Short Signals: {(tqqq_feat_pred['Sig'] == -1).sum()}")
    
    print("Running 5m Path Simulation (Trailing ATR)...")
    
    tqqq_feat_pred['datetime'] = pd.to_datetime(tqqq_feat_pred['datetime'])
    master_5m['datetime'] = pd.to_datetime(master_5m['datetime'])
    
    df_merged = pd.merge_asof(master_5m.sort_values('datetime'), 
                              tqqq_feat_pred[['datetime', 'Sig']].sort_values('datetime'),
                              on='datetime', direction='backward')
                              
    df_merged['Sig'] = df_merged['Sig'].fillna(0)
    
    capital = 10_000_000.0
    pos = 0
    entry_px = 0
    peak_px = 0
    cap_curve = []
    dates = []
    current_sym = None
    
    long_trades = 0
    short_trades = 0
    
    df_merged['ATR'] = df_merged['High'] - df_merged['Low']
    df_merged['ATR_14'] = df_merged['ATR'].rolling(14).mean().fillna(df_merged['ATR'].mean())
    
    for row in df_merged.itertuples():
        sig = row.Sig
        
        if row.datetime.hour == 15 and row.datetime.minute >= 50 and pos > 0:
            if current_sym == 'TQQQ': last_px = row.Close
            else: last_px = row.SQQQ_Close
            
            if not pd.isna(last_px):
                capital += pos * last_px * (1 - 0.0020)
                pos = 0
                current_sym = None
                cap_curve.append(capital)
                dates.append(row.datetime)
            continue
            
        if pos == 0:
            if sig == 1:
                current_sym = 'TQQQ'
                cur_px = row.Close
                if pd.isna(cur_px): continue
                pos = (capital * 0.98) / cur_px
                entry_px = cur_px
                peak_px = entry_px
                capital -= pos * entry_px
                long_trades += 1
            elif sig == -1:
                current_sym = 'SQQQ'
                cur_px = row.SQQQ_Close
                if pd.isna(cur_px): continue
                pos = (capital * 0.98) / cur_px
                entry_px = cur_px
                peak_px = entry_px
                capital -= pos * entry_px
                short_trades += 1
        elif pos > 0:
            if current_sym == 'TQQQ': cur_px = row.Close
            else: cur_px = row.SQQQ_Close
                
            if pd.isna(cur_px): continue
            if cur_px > peak_px: peak_px = cur_px
            
            ret = (cur_px / entry_px) - 1.0
            max_ret = (peak_px / entry_px) - 1.0
            
            sl_pct = max(0.02, min(0.032, (row.ATR_14 * 1.5) / cur_px))
            
            sell = False
            if max_ret >= 0.015 and cur_px <= peak_px * (1 - 0.003): sell = True
            elif ret >= 0.10: sell = True
            elif ret <= -sl_pct: sell = True
            
            if sell:
                capital += pos * cur_px * (1 - 0.0020)
                pos = 0
                current_sym = None
                cap_curve.append(capital)
                dates.append(row.datetime)
                
    if pos > 0:
        if current_sym == 'TQQQ': last_px = df_merged.iloc[-1]['Close']
        else: last_px = df_merged.iloc[-1]['SQQQ_Close']
        capital += pos * last_px * (1 - 0.0020)
        cap_curve.append(capital)
        dates.append(df_merged.iloc[-1]['datetime'])
        
    mdd = 0
    if len(cap_curve) > 0:
        cc = np.array(cap_curve)
        rm = np.maximum.accumulate(cc)
        mdd = np.max((rm - cc) / rm) * 100.0
        
    test_start = tqqq_feat_pred['datetime'].min()
    test_end = tqqq_feat_pred['datetime'].max()
    test_years = (test_end - test_start).days / 365.25
    cagr = ((capital / 10000000.0) ** (1 / test_years) - 1.0) * 100
        
    print(f"\n--- Results ---")
    print(f"Test Period: {test_start.strftime('%Y-%m-%d')} ~ {test_end.strftime('%Y-%m-%d')} ({test_years:.1f} years)")
    print(f"Final Capital: {capital:,.0f} KRW")
    print(f"Total Return: {((capital / 10000000.0) - 1.0) * 100:.2f}%")
    print(f"CAGR: {cagr:.2f}%")
    print(f"MDD: {mdd:.2f}%")
    print(f"Long Trades: {long_trades}, Short Trades: {short_trades}")
    
    plt.figure(figsize=(14, 7))
    if len(cap_curve) > 0:
        path_df = pd.DataFrame({'datetime': dates, 'capital': cap_curve})
        path_df = path_df.set_index('datetime')
        path_df = path_df[~path_df.index.duplicated(keep='last')]
        
        plt.plot(path_df.index, path_df['capital'], label='Real Data WFA', color='blue')
        plt.title('100% Real Data WFA Backtest (Log Scale)')
        plt.yscale('log')
        plt.xlabel('Date')
        plt.ylabel('Capital (KRW)')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig('data/real_data_chart.png')
        print("Chart saved to data/real_data_chart.png")

if __name__ == '__main__':
    run_real_backtest()
