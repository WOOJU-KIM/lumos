import re

with open('scripts/run_weekly_rolling_wfa.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace ROLLING_WINDOW_WEEKS at the top
content = re.sub(
    r"ROLLING_WINDOW_WEEKS = 104.*?if len\(unique_weeks\) <= ROLLING_WINDOW_WEEKS:\n\s+raise RuntimeError\([^)]+\)",
    "MIN_TRAIN_WEEKS = 104\n    MAX_TRAIN_WEEKS = 520\n    if len(unique_weeks) <= MIN_TRAIN_WEEKS:\n        raise RuntimeError('데이터 주차가 104주보다 적습니다.')",
    content,
    flags=re.DOTALL
)

# Replace test_start_idx
content = content.replace("test_start_idx = ROLLING_WINDOW_WEEKS", "test_start_idx = MIN_TRAIN_WEEKS")

# Replace loop slicing
old_loop_code = """
        cur_test_week = unique_weeks[w_idx]
        train_weeks = unique_weeks[w_idx - ROLLING_WINDOW_WEEKS : w_idx]
        
        # A. 직전 104주(2년) 학습 데이터 슬라이싱
"""
new_loop_code = """
        cur_test_week = unique_weeks[w_idx]
        train_start_idx = max(0, w_idx - MAX_TRAIN_WEEKS)
        train_weeks = unique_weeks[train_start_idx : w_idx]
        
        # A. 최대 10년 누적 학습 데이터 슬라이싱
"""
content = content.replace(old_loop_code.strip(), new_loop_code.strip())
content = content.replace("'rolling_window_weeks': ROLLING_WINDOW_WEEKS,", "'rolling_window_weeks': MAX_TRAIN_WEEKS,")

with open('scripts/temp_run_wfa_10yr.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Rewrote temp_run_wfa_10yr.py cleanly")
