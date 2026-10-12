import re

with open('core/ml_engine.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Add _safe_merge_asof function
helper = '''
def _safe_merge_asof(df_left: pd.DataFrame, df_right: pd.DataFrame, on='datetime_dt', direction='backward') -> pd.DataFrame:
    """Overlapping columns drop & merge_asof utility"""
    cols_to_drop = [c for c in df_right.columns if c in df_left.columns and c != on]
    if cols_to_drop:
        df_left = df_left.drop(columns=cols_to_drop)
    return pd.merge_asof(df_left.sort_values(on), df_right.sort_values(on), on=on, direction=direction)

class MLFeatureEngine:'''

text = text.replace('class MLFeatureEngine:', helper)

# 2. Refactor soxx_feat merge
old_soxx_merge = r"""                cols_to_drop = \[c for c in soxx_feat\.columns if c in df\.columns and c != 'datetime_dt'\]\n                if cols_to_drop:\n                    df = df\.drop\(columns=cols_to_drop\)\n                df = pd\.merge_asof\(df\.sort_values\('datetime_dt'\), \n                                   soxx_feat\.sort_values\('datetime_dt'\)\.drop\(columns=\['datetime'\]\), \n                                   on='datetime_dt', direction='backward'\)"""
new_soxx_merge = "                df = _safe_merge_asof(df, soxx_feat.drop(columns=['datetime']), on='datetime_dt', direction='backward')"
text = re.sub(old_soxx_merge, new_soxx_merge, text)

# 3. Refactor cross asset merge
old_cross_merge = r"""                    cols_to_drop = \[c for c in cols_to_merge if c in df\.columns and c != 'datetime_dt'\]\n                    if cols_to_drop:\n                        df = df\.drop\(columns=cols_to_drop\)\n                    df = pd\.merge_asof\(df\.sort_values\('datetime_dt'\), \n                                       sym_raw\[cols_to_merge\]\.sort_values\('datetime_dt'\), \n                                       on='datetime_dt', direction='backward'\)"""
new_cross_merge = "                    df = _safe_merge_asof(df, sym_raw[cols_to_merge], on='datetime_dt', direction='backward')"
text = re.sub(old_cross_merge, new_cross_merge, text)

# 4. Make sym prefix generic for BASE_ASSET_SHORT
old_sym_lower = r"""                    lower_sym = sym\.lower\(\)"""
new_sym_lower = "                    lower_sym = 'base_short' if sym == config.BASE_ASSET_SHORT else sym.lower()"
text = re.sub(old_sym_lower, new_sym_lower, text)

# 5. Make soxl explicitly base_long
text = text.replace("df['soxl_ret_20']", "df['base_long_ret_20']")
text = text.replace("df['soxl_ret_5']", "df['base_long_ret_5']")
text = text.replace("df['soxl_vs_qqq_20']", "df['base_long_vs_qqq_20']")
text = text.replace("df['soxl_vs_qqq_5']", "df['base_long_vs_qqq_5']")
text = text.replace("df.get('soxl_ret_20'", "df.get('base_long_ret_20'")
text = text.replace("df.get('soxl_ret_5'", "df.get('base_long_ret_5'")

with open('core/ml_engine.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Refactoring complete')
