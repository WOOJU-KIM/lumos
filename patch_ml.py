import re
from pathlib import Path

p = Path('core/ml_engine.py')
c = p.read_text(encoding='utf-8', errors='ignore')

# We need to find the return 0.50 statements. Let's just use regex
c = re.sub(
    r'(if df_candle_15m is None or df_candle_15m\.empty or len\(df_candle_15m\) < 15:\s+)return 0, 0\.50, "(.*?)"',
    r'\1print(f"DEBUG ml_engine: df_candle_15m is invalid. len: {len(df_candle_15m) if df_candle_15m is not None else 0}")\n        return 0, 0.50, "\2"',
    c
)

c = re.sub(
    r'(if df_feat\.empty:\s+)return 0, 0\.50, "(.*?)"',
    r'\1print("DEBUG ml_engine: df_feat is empty after extract_features")\n        return 0, 0.50, "\2"',
    c
)

p.write_text(c, encoding='utf-8')
print('Patched ml_engine.py')
