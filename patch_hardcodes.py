import re
with open('config.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove duplicate GBDT_CONFIDENCE_THRESHOLD (line 20)
content = re.sub(r'GBDT_CONFIDENCE_THRESHOLD\s*=\s*0\.704\s*#\s*GBDT\s*.*?\(65%\)\n', '', content)

# Add fallback price and calibration constants
additions = """
# 8. 모델 및 캘리브레이션 파라미터 (Model Calibration)
FALLBACK_PRICE = 120.74
CALIB_BASE = 0.50
CALIB_PIVOT = 0.333
CALIB_SCALE = 1.15
"""
content += additions

with open('config.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Patch ml_engine.py
with open('core/ml_engine.py', 'r', encoding='utf-8') as f:
    ml = f.read()

ml = ml.replace("0.50 + (pl - 0.333) * 1.15", "config.CALIB_BASE + (pl - config.CALIB_PIVOT) * config.CALIB_SCALE")
ml = ml.replace("0.50 + (ps - 0.333) * 1.15", "config.CALIB_BASE + (ps - config.CALIB_PIVOT) * config.CALIB_SCALE")
ml = ml.replace("0.50 + (pn - 0.333) * 1.15", "config.CALIB_BASE + (pn - config.CALIB_PIVOT) * config.CALIB_SCALE")
ml = ml.replace("max(0.50,", "max(config.CALIB_BASE,")

# Also use GBDT_FEATURE_FRACTION
ml = ml.replace("subsample=0.8,", "subsample=config.GBDT_FEATURE_FRACTION,")
ml = ml.replace("colsample_bytree=0.8,", "colsample_bytree=config.GBDT_FEATURE_FRACTION,")
# For the one I patched earlier:
ml = ml.replace("deterministic=True,", "subsample=config.GBDT_FEATURE_FRACTION,\n                colsample_bytree=config.GBDT_FEATURE_FRACTION,\n                deterministic=True,")

with open('core/ml_engine.py', 'w', encoding='utf-8') as f:
    f.write(ml)

# Patch live_runner.py (Timeout and Slippage)
with open('core/live_runner.py', 'r', encoding='utf-8') as f:
    lr = f.read()

lr = lr.replace("ORDER_TIMEOUT_BUY_SEC: float = 3.0", "ORDER_TIMEOUT_BUY_SEC: float = config.BUY_TIMEOUT_SEC")
lr = lr.replace("ORDER_TIMEOUT_SELL_SEC: float = 3.0", "ORDER_TIMEOUT_SELL_SEC: float = config.SELL_TIMEOUT_SEC")
lr = lr.replace("order_px_1 = round(cur_px + 0.03, 2)", "order_px_1 = round(cur_px + config.BUY_SLIPPAGE_ADJUST, 2)")
lr = lr.replace("order_px_1 = round(cur_px - 0.05, 2)", "order_px_1 = round(cur_px - config.SELL_SLIPPAGE_ADJUST, 2)")
lr = lr.replace("order_px_2 = round(cur_px + 0.03, 2)", "order_px_2 = round(cur_px + config.BUY_SLIPPAGE_ADJUST, 2)")
lr = lr.replace("order_px_2 = round(cur_px - 0.05, 2)", "order_px_2 = round(cur_px - config.SELL_SLIPPAGE_ADJUST, 2)")
lr = lr.replace("timeout_sec = float(os.getenv('ORDER_TIMEOUT', 3.0))", "timeout_sec = float(os.getenv('ORDER_TIMEOUT', config.BUY_TIMEOUT_SEC))")

with open('core/live_runner.py', 'w', encoding='utf-8') as f:
    f.write(lr)

# Patch brokers
for broker_file in ['core/hybrid_broker.py', 'core/kiwoom_broker.py']:
    with open(broker_file, 'r', encoding='utf-8') as f:
        br = f.read()
    br = br.replace("120.74", "config.FALLBACK_PRICE")
    with open(broker_file, 'w', encoding='utf-8') as f:
        f.write(br)
