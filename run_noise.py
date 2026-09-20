import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from core.data_lake import MarketDataLake
import config
from core.ml_engine import MLFeatureEngine
import joblib
from pathlib import Path
import time

def run_noise_mc():
    print("Loading base data...")
    lake = MarketDataLake()
    tqqq_15m_base = lake.load_rolling_candles('TQQQ', '15m', max_trading_days=1000)
    tqqq_5m_base = lake.load_rolling_candles('TQQQ', '5m', max_trading_days=1000)
    
    tqqq_15m_base['datetime'] = pd.to_datetime(tqqq_15m_base['datetime'])
    tqqq_5m_base['datetime'] = pd.to_datetime(tqqq_5m_base['datetime'])

    n_sims = 10
    caps = []
    mdds = []
    
    model_path = Path('models') / 'model_main_data_refresh.pkl'
    clf = joblib.load(model_path)
    
    ml_engine = MLFeatureEngine(confidence_threshold=0.60)
    ml_engine.model = clf
    ml_engine.is_trained = True
    
    print(f"Running {n_sims} noise simulations...")
    start_all = time.time()
    
    for i in range(n_sims):
        s_time = time.time()
        tqqq_15m = tqqq_15m_base.copy()
        tqqq_5m = tqqq_5m_base.copy()
        
        noise15 = np.random.uniform(-0.01, 0.01, size=len(tqqq_15m))
        noise5 = np.random.uniform(-0.01, 0.01, size=len(tqqq_5m))
        
        for col in ['Open', 'High', 'Low', 'Close']:
            tqqq_15m[col] = tqqq_15m[col] * (1.0 + noise15)
            tqqq_5m[col] = tqqq_5m[col] * (1.0 + noise5)
            
        tqqq_feat = ml_engine.extract_features(tqqq_15m)
        tqqq_feat = ml_engine.add_confidence_columns(tqqq_feat)
        
        tqqq_feat['Sig'] = 0
        conf_mask = tqqq_feat['Prob_Long'] >= 0.60
        tqqq_feat.loc[conf_mask, 'Sig'] = 1
        
        df_merged = pd.merge_asof(tqqq_5m.sort_values('datetime'), 
                                  tqqq_feat.reset_index(drop=True)[['datetime', 'Sig', 'Prob_Long']].sort_values('datetime'),
                                  on='datetime', direction='backward')
        
        capital = 10_000_000.0
        pos = 0
        entry_px = 0
        peak_px = 0
        cap_curve = [capital]
        
        for row in df_merged.itertuples():
            sig = row.Sig
            cur_px = row.Close
            
            if pos == 0 and sig == 1:
                pos = (capital * 0.98) / cur_px
                entry_px = cur_px
                peak_px = entry_px
                capital -= pos * entry_px
            elif pos > 0:
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
                    cap_curve.append(capital)
                    
        if pos > 0:
            capital += pos * df_merged.iloc[-1]['Close'] * (1 - 0.0020)
            cap_curve.append(capital)
            
        mdd = 0
        if len(cap_curve) > 0:
            cc = np.array(cap_curve)
            rm = np.maximum.accumulate(cc)
            mdd = np.max((rm - cc) / rm) * 100.0
            
        caps.append(capital)
        mdds.append(mdd)
        print(f'Sim {i+1}: Cap={capital:,.0f} KRW, MDD={mdd:.2f}%, Time={time.time()-s_time:.2f}s')
        
    print('=== 2. OHLC Noise Backtest (-1% to +1% Random Noise) ===')
    print(f'Simulations: {n_sims}')
    print(f'Mean Final Capital: {np.mean(caps):,.0f} KRW')
    print(f'Worst Final Capital: {np.min(caps):,.0f} KRW')
    print(f'Mean MDD: {np.mean(mdds):.2f}%')
    print(f'Worst MDD: {np.max(mdds):.2f}%')
    print(f'Total Time: {time.time()-start_all:.2f}s')

if __name__ == '__main__':
    run_noise_mc()
