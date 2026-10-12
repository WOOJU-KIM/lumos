import sys
import re

with open("core/ml_engine.py", "r", encoding="cp949", errors="replace") as f:
    core_code = f.read()

with open("scripts/temp_ml_engine_sandbox.py", "r", encoding="utf-8") as f:
    sandbox_code = f.read()

# Extract from sandbox
match_extract = re.search(r"    def extract_features\(self, df: pd.DataFrame, live_prices: dict = None\) -> pd.DataFrame:.*?(?=    def train_and_select_top_features)", sandbox_code, re.DOTALL)
if not match_extract:
    print("Failed to find extract_features in sandbox")
    sys.exit(1)
sandbox_extract = match_extract.group(0)

# 2. Replace extract_features in core
core_code = re.sub(r"    def extract_features\(self, df: pd.DataFrame, live_prices: dict = None\) -> pd.DataFrame:.*?(?=    @staticmethod\n    def compute_triple_barrier_labels)", sandbox_extract, core_code, flags=re.DOTALL)

# 3. Modify compute_triple_barrier_labels
core_code = re.sub(r"take_profit: float = config.MAX_TP_PCT,", "take_profit: float = 0.030,", core_code)
core_code = re.sub(r"stop_loss: float = config.SL_MIN_PCT,", "stop_loss: float = 0.020,", core_code)
core_code = re.sub(r"horizon: int = int\(config.TIME_STOP_MINUTES / 15\)", "horizon: int = 6", core_code)

# 4. Modify train_and_select_top_features
nan_handling = """
        # [NEW] NaN 과다 컬럼 제외 및 보간
        threshold = len(feat_df) * 0.3
        valid_cols = []
        for c in feature_cols:
            if feat_df[c].isna().sum() < threshold:
                valid_cols.append(c)
        feature_cols = valid_cols
        feat_df[feature_cols] = feat_df[feature_cols].fillna(0.0)
        
        clean_data = feat_df[feature_cols + ['Target']].dropna()
        
        # [NEW] 10년 롤링 윈도우 (과거 10년 데이터만 학습에 사용)
        if isinstance(clean_data.index, pd.DatetimeIndex) and len(clean_data) > 0:
            import pandas as pd
            cutoff_10y = clean_data.index.max() - pd.DateOffset(years=10)
            clean_data = clean_data[clean_data.index >= cutoff_10y]
            
        if len(clean_data) < 100:
"""
core_code = re.sub(r"        clean_data = feat_df\[feature_cols \+ \['Target'\]\]\.dropna\(\)\n        if len\(clean_data\) < 100:", nan_handling, core_code)

# Remove any print statements
core_code = core_code.replace("print(\"[Feature Engineering]", "logger.debug(\"[Feature Engineering]")
core_code = core_code.replace("print(f\"[SandboxedMLFeatureEngine]", "logger.warning(f\"[MLFeatureEngine]")

with open("core/ml_engine.py", "w", encoding="cp949", errors="replace") as f:
    f.write(core_code)

print("Patch applied successfully.")
