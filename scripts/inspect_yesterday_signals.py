import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from core.data_lake import MarketDataLake
from core.moe_orchestrator import MoEMetaOrchestrator
from core.heterogeneous_models import CrossAssetDislocationModel
from core.ml_engine import MLFeatureEngine

lake = MarketDataLake()
long_15m = lake.load_candles(config.BASE_ASSET_LONG, "15m")
nvda_15m = lake.load_candles("NVDA", "15m")
qqq_15m = lake.load_candles("QQQ", "15m")
vix_15m = lake.load_candles("^VIX", "15m")
macro_60m = lake.load_candles(config.MACRO_TREND_SYMBOL, "60m")

print("=== TQQQ 최근 5개 15분봉 ===")
print(long_15m[["Open", "High", "Low", "Close", "Volume"]].tail(5))

c = long_15m["Close"]
ret_5 = float((c.iloc[-1] / c.iloc[-5] - 1.0))
print(f"\nTQQQ 5봉 수익률: {ret_5*100:.2f}%")

# 1. 크로스에셋 결과
cross_model = CrossAssetDislocationModel()
n_c = nvda_15m["Close"]
q_c = qqq_15m["Close"]
v_c = vix_15m["Close"]
macro_15m = lake.load_candles(config.MACRO_TREND_SYMBOL, "15m")
sx_c = macro_15m["Close"] if not macro_15m.empty else n_c
macro_r = float(sx_c.iloc[-1] / sx_c.iloc[-5] - 1.0) if len(sx_c) >= 5 else nvda_r

sig_code, exp_conf, meta = cross_model.predict_signal(
    long_ret=ret_5,
    nvda_ret=nvda_r,
    macro_ret=macro_r,
    qqq_ret=qqq_r,
    vix_ret=vix_r,
    tnx_ret=0.0
)
print(f"\n[크로스에셋 괴리 모델] sig_code={sig_code}, exp_conf={exp_conf}")
print("Meta:", meta)

# 2. GBDT 결과
gbdt = MLFeatureEngine()
df_feat = gbdt.extract_features(long_15m)
last_feat = df_feat.iloc[-1]
print(f"\n[GBDT 모델 피처]")
print(f"Confidence: {last_feat.get('Confidence')}")
print(f"VWAP_Diff: {last_feat.get('VWAP_Diff')}")
print(f"RSI_14: {last_feat.get('RSI_14')}")
print(f"BB_Lower: {last_feat.get('BB_Lower')}")
print(f"Close: {last_feat.get('Close')}")

# 3. 60분봉 추세
if not macro_60m.empty:
    macro_ema20 = macro_60m['Close'].ewm(span=20, adjust=False).mean().iloc[-1]
    macro_c = macro_60m['Close'].iloc[-1]
    print(f"\n[60분봉 추세] SOXX Close: {macro_c:.2f}, EMA20: {macro_ema20:.2f} (차이: {((macro_c/macro_ema20)-1)*100:.2f}%)")

moe = MoEMetaOrchestrator(confidence_threshold=0.75)
res = moe.evaluate_dual_filter_signal(long_15m, threshold=0.75)
print("\n=== 최종 MoE 평가 결과 ===")
print(json.dumps(res, indent=2, ensure_ascii=False))
