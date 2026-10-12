import pandas as pd

with open('scripts/run_weekly_rolling_wfa.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip_mode = False
for i, line in enumerate(lines):
    if "train_mask = long_feat['week_id'].isin(train_weeks)" in line:
        new_lines.append(line)
        new_lines.append("        # 실전(Live)과 100% 동일하게 MLFeatureEngine 호출\n")
        new_lines.append("        max_train_dt = long_feat.loc[train_mask, 'datetime_dt'].max()\n")
        new_lines.append("        train_df = long_15m[pd.to_datetime(long_15m['datetime']) <= max_train_dt]\n")
        new_lines.append("        ml_live = MLFeatureEngine(confidence_threshold=config.GBDT_CONFIDENCE_THRESHOLD)\n")
        new_lines.append("        live_df = pd.read_csv('data/live_trades.csv')\n")
        new_lines.append("        live_df = live_df[pd.to_datetime(live_df['entry_time']) <= max_train_dt]\n")
        new_lines.append("        clf, _, _ = ml_live.train_and_select_top_features(train_df, live_experience_df=live_df)\n")
        skip_mode = True
        continue
    
    if skip_mode and 'clf.fit' in line:
        skip_mode = False
        continue
        
    if skip_mode:
        continue
        
    if 'X_test = test_df[feature_cols].fillna(0.0)' in line:
        new_lines.append("        X_test = test_df[clf.feature_name_].fillna(0.0)\n")
        continue
        
    if 'time_str < config.PHASE_MAIN_END' in line:
        new_lines.append(line.replace('time_str < config.PHASE_MAIN_END', 'config.PHASE_MAIN_START <= time_str < config.PHASE_MAIN_END'))
        continue

    new_lines.append(line)

with open('scripts/run_weekly_rolling_wfa_live_sync.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
