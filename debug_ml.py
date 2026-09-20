import pandas as pd
import sqlite3
from config import DATA_DIR
from core.moe_orchestrator import MoEMetaOrchestrator
from core.ml_engine import MLFeatureEngine

# Apply the predict_signal monkey patch
orig_predict_signal = MLFeatureEngine.predict_signal
def patched_predict_signal(self, df_candle_15m, **kwargs):
    return orig_predict_signal(self, df_candle_15m)
MLFeatureEngine.predict_signal = patched_predict_signal

conn = sqlite3.connect(DATA_DIR / 'market_data.db')
df = pd.read_sql("SELECT * FROM market_candles WHERE symbol='TQQQ' AND timeframe='15m' ORDER BY datetime ASC", conn)
df['datetime'] = pd.to_datetime(df['datetime'])
df.rename(columns={'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'}, inplace=True)
train = df.iloc[-13000:-130]
test = df.iloc[-130:]
moe = MoEMetaOrchestrator(confidence_threshold=0.60, mode='hybrid_v3')
print('training')
moe.gbdt_engine.train_and_select_top_features(train)
print('evaluating')
res = moe.evaluate_dual_filter_signal(test.iloc[:60], current_time_str=str(test['datetime'].iloc[59]))
print(res)

