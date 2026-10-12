import pandas as pd
import numpy as np

from core.market_data_lake import MarketDataLake
import config
from core.ml_engine import MLFeatureEngine

def run_noise_mc():
    lake = MarketDataLake()
    
    # Base load to get date range
    long_15m_base = lake.load_rolling_candles(config.BASE_ASSET_LONG, '15m', max_trading_days=1000)
    long_5m_base = lake.load_rolling_candles(config.BASE_ASSET_LONG, '5m', max_trading_days=1000)
    macro_15m_base = lake.load_rolling_candles(config.MACRO_TREND_SYMBOL, '15m', max_trading_days=1000)
    nvda_15m_base = lake.load_rolling_candles('NVDA', '15m', max_trading_days=1000)
    vix_15m_base = lake.load_rolling_candles('VIX', '15m', max_trading_days=1000)

    n_sims = 20
    caps = []
    mdds = []
    
    for i in range(n_sims):
        # Create copies and add noise
        long_15m = long_15m_base.copy()
        long_5m = long_5m_base.copy()
        
        noise15 = np.random.uniform(-0.01, 0.01, size=len(long_15m))
        noise5 = np.random.uniform(-0.01, 0.01, size=len(long_5m))
        
        for col in ['Open', 'High', 'Low', 'Close']:
            long_15m[col] = long_15m[col] * (1.0 + noise15)
            long_5m[col] = long_5m[col] * (1.0 + noise5)
            
        # Overwrite lake method for this run
        def mock_load(symbol, tf, max_trading_days=1000, start_date=None, end_date=None):
            if symbol == config.BASE_ASSET_LONG and tf == '15m': return long_15m.copy()
            if symbol == config.BASE_ASSET_LONG and tf == '5m': return long_5m.copy()
            if symbol == config.MACRO_TREND_SYMBOL and tf == '15m': return macro_15m_base.copy()
            if symbol == 'NVDA' and tf == '15m': return nvda_15m_base.copy()
            if symbol == 'VIX' and tf == '15m': return vix_15m_base.copy()
            return pd.DataFrame()
            
        lake.load_rolling_candles = mock_load
        
        # Manually invoke backtest logic
        # It's faster to just use the pre-trained model for 22_24
        import joblib
        from pathlib import Path
        model_path = Path("models") / "model_main_data_refresh.pkl"
        clf = joblib.load(model_path)
        
        ml_engine = MLFeatureEngine(confidence_threshold=0.60)
        ml_engine.model = clf
        ml_engine.is_trained = True
        
        long_feat = ml_engine.extract_features(long_15m)
        long_feat = ml_engine.add_confidence_columns(long_feat)
        
        long_feat['Sig'] = 0
        conf_mask = long_feat['Prob_Long'] >= 0.60
        long_feat.loc[conf_mask, 'Sig'] = 1
        
        # Merge with 5m
        df_merged = pd.merge_asof(long_5m.sort_values('datetime'), 
                                  long_feat[['datetime', 'Sig', 'Prob_Long']].sort_values('datetime'),
                                  on='datetime', direction='backward')
        
        # Basic 5m backtest loop
        capital = 10_000_000.0
        pos = 0
        entry_px = 0
        peak_px = 0
        
        cap_curve = [capital]
        
        for idx, row in df_merged.iterrows():
            if pos == 0 and row['Sig'] == 1:
                pos = (capital * 0.98) / row['Close']
                entry_px = row['Close']
                peak_px = entry_px
                capital -= pos * entry_px
            elif pos > 0:
                cur_px = row['Close']
                if cur_px > peak_px: peak_px = cur_px
                
                # Check TS / SL
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
        
    print("=== 2. OHLC Noise Backtest (-1% to +1% Random Noise) ===")
    print(f"Simulations: {n_sims}")
    print(f"Mean Final Capital: {np.mean(caps):,.0f} KRW")
    print(f"Worst Final Capital: {np.min(caps):,.0f} KRW")
    print(f"Mean MDD: {np.mean(mdds):.2f}%")
    print(f"Worst MDD: {np.max(mdds):.2f}%")

if __name__ == '__main__':
    run_noise_mc()
