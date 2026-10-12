import re

with open('core/ml_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add get_feature_columns method
get_feat_cols_method = '''
    @staticmethod
    def get_feature_columns(df: pd.DataFrame) -> list:
        exclude_cols = {
            'open', 'high', 'low', 'close', 'volume', 
            'Open', 'High', 'Low', 'Close', 'Volume',
            'datetime', 'datetime_dt', 'date_str', 'time_str', 'date',
            'week_id', 'target', 'Target',
            'cum_vp', 'vwap', 'EMA_9', 'EMA_21', 'EMA_50', 'EMA_200'
        }
        return [c for c in df.columns if c not in exclude_cols and not str(c).startswith('week_') and pd.api.types.is_numeric_dtype(df[c])]
'''

# Insert it before extract_features or train_and_select_top_features
# I will just replace `def train_and_select_top_features` with the method + `def train_and_select_top_features`

content = content.replace('    def train_and_select_top_features', get_feat_cols_method + '\n    def train_and_select_top_features')

# Now replace feature_cols = ... with feature_cols = self.get_feature_columns(feat_df)
old_feature_cols = "        feature_cols = [c for c in feat_df.columns if c not in ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'cum_vp', 'vwap', 'EMA_9', 'EMA_21', 'EMA_50', 'EMA_200'] and not c.startswith('week_') and pd.api.types.is_numeric_dtype(feat_df[c])]"
new_feature_cols = "        feature_cols = self.get_feature_columns(feat_df)"
content = content.replace(old_feature_cols, new_feature_cols)

with open('core/ml_engine.py', 'w', encoding='utf-8') as f:
    f.write(content)
