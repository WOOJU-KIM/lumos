import re

with open('scripts/run_weekly_rolling_wfa.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_feature_logic = """    # ?처 컬럼 ?별
    exclude_cols = {'open', 'high', 'low', 'close', 'volume', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'datetime', 'datetime_dt', 'date_str', 'time_str', 'week_id', 'target', 'date'}
    feature_cols = [c for c in long_feat.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(long_feat[c])]
    print(f"? ?습 ?처 ?? {len(feature_cols)}??처")"""

new_feature_logic = """    # 피처 컬럼 식별
    feature_cols = MLFeatureEngine.get_feature_columns(long_feat)
    print(f"총 학습 피처 수: {len(feature_cols)}개 피처")"""

# Regex patch
pattern = re.compile(
    r"\s+# [^\n]+\n"
    r"\s+exclude_cols = \{[^\}]+\}\n"
    r"\s+feature_cols = \[c for c in long_feat\.columns[^\n]+\n"
    r"\s+print\(f\"[^\n]+\)"
)
content = pattern.sub('\n' + new_feature_logic, content)

with open('scripts/run_weekly_rolling_wfa.py', 'w', encoding='utf-8') as f:
    f.write(content)
