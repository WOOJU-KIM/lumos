with open('scripts/temp_run_wfa_10yr.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace("train_weeks = unique_weeks[w_idx - ROLLING_WINDOW_WEEKS : w_idx]", "")
content = content.replace("'rolling_window_weeks': ROLLING_WINDOW_WEEKS,", "'rolling_window_weeks': MAX_TRAIN_WEEKS,")

with open('scripts/temp_run_wfa_10yr.py', 'w', encoding='utf-8') as f:
    f.write(content)
