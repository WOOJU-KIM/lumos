import traceback
from core.ml_engine import MLFeatureEngine
from core.data_lake import MarketDataLake
import pandas as pd

def test_gbdt():
    print("Initializing MLFeatureEngine...")
    engine = MLFeatureEngine()
    print(f"Model loaded: {engine.model is not None}")
    print(f"Number of features required: {len(engine.feature_names)}")
    
    lake = MarketDataLake()
    df = lake.load_candles("TQQQ", "15m")
    
    if df.empty:
        print("Could not load TQQQ 15m candles from data lake, generating dummy data.")
        # generate dummy df
        import numpy as np
        dates = pd.date_range("2026-09-01", periods=100, freq="15T")
        df = pd.DataFrame({
            "datetime": dates,
            "Open": np.random.uniform(20, 30, 100),
            "High": np.random.uniform(25, 35, 100),
            "Low": np.random.uniform(15, 25, 100),
            "Close": np.random.uniform(20, 30, 100),
            "Volume": np.random.randint(1000, 10000, 100)
        })
    else:
        print(f"Loaded {len(df)} candles for TQQQ")
        df = df.tail(100)
    
    print("Extracting features...")
    feat_df = engine.extract_features(df)
    print(f"Feature dataframe size: {feat_df.shape}")
    
    print("Adding confidence columns...")
    try:
        res = engine.add_confidence_columns(feat_df)
        print("Top 5 Confidences:")
        print(res[['datetime_dt', 'Signal', 'Confidence', 'Prob_Long', 'Prob_Short', 'Prob_Neutral']].tail())
    except Exception as e:
        print(f"Exception during add_confidence_columns: {e}")
        traceback.print_exc()

    print("\nTesting predict_signal interface...")
    sig, conf, reason = engine.predict_signal(df)
    print(f"Signal: {sig}")
    print(f"Confidence: {conf}")
    print(f"Reason: {reason}")

if __name__ == "__main__":
    test_gbdt()
