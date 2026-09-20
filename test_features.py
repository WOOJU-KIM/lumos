from core.data_lake import MarketDataLake
from core.ml_engine import MLFeatureEngine
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

dl = MarketDataLake()
df = dl.load_candles("TQQQ", "15m")

ml = MLFeatureEngine()
df_feat = ml.extract_features(df)
v3_features = ['Trend_Duration', 'Trend_Consistency', 'RSI_Adj', 'Directional_Squeeze', 'Smart_Money_Flow']

print(df_feat[v3_features].tail(3))
