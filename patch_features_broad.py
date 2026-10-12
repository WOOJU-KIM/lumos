import re

with open('core/ml_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_func = '''    def get_feature_columns(df: pd.DataFrame) -> list:
        exclude_cols = {
            'open', 'high', 'low', 'close', 'volume', 
            'Open', 'High', 'Low', 'Close', 'Volume',
            'datetime', 'datetime_dt', 'date_str', 'time_str', 'date',
            'week_id', 'target', 'Target',
            'cum_vp', 'vwap', 'EMA_9', 'EMA_21', 'EMA_50', 'EMA_200'
        }
        return [c for c in df.columns if c not in exclude_cols and not str(c).startswith('week_') and pd.api.types.is_numeric_dtype(df[c])]'''

new_func = '''    def get_feature_columns(df: pd.DataFrame) -> list:
        exclude_cols = {
            'open', 'high', 'low', 'close', 'volume', 
            'Open', 'High', 'Low', 'Close', 'Volume',
            'datetime', 'datetime_dt', 'date_str', 'time_str', 'date', 'Datetime',
            'week_id', 'target', 'Target',
            'cum_vp', 'vwap', 'EMA_9', 'EMA_21', 'EMA_50', 'EMA_200',
            'Confidence', 'Signal', 'Direction', 'Prob_Long', 'Prob_Short', 'Prob_Neutral', 'cross_dir'
        }
        return [c for c in df.columns if c not in exclude_cols and not str(c).startswith('week_') and pd.api.types.is_numeric_dtype(df[c])]'''

content = content.replace(old_func, new_func)

with open('core/ml_engine.py', 'w', encoding='utf-8') as f:
    f.write(content)

with open('core/weekly_tournament.py', 'r', encoding='utf-8') as f:
    content = f.read()

pattern = re.compile(
    r"\s+feature_cols = \[c for c in df\.columns if c not in \[[^\]]+\] and pd\.api\.types\.is_numeric_dtype\(df\[c\]\)\]"
)
content = pattern.sub('\n        feature_cols = MLFeatureEngine.get_feature_columns(df)', content)

with open('core/weekly_tournament.py', 'w', encoding='utf-8') as f:
    f.write(content)
