import sys
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from core.moe_orchestrator import train_and_save_moe_orchestrator, MOE_MODEL_PATH
from core.data_lake import MarketDataLake

def update_model():
    print("Fetching data from DB...")
    lake = MarketDataLake()
    df_15m = lake.load_rolling_candles(config.BASE_ASSET_LONG, "15m", max_trading_days=504)
    print(f"Loaded {len(df_15m)} 15m candles for {config.BASE_ASSET_LONG}.")
    
    print(f"Training final LIVE model with config GBDT_CONF={config.GBDT_CONFIDENCE_THRESHOLD}...")
    moe = train_and_save_moe_orchestrator(
        confidence_threshold=config.GBDT_CONFIDENCE_THRESHOLD, 
        mode="hybrid_v3"
    )
    print("Model saved to:", MOE_MODEL_PATH)
    
if __name__ == "__main__":
    update_model()
