import os
import sys
with open('c:\\Users\\chabo\\OneDrive\\바탕 화면\\lumos\\scripts\\run_weekly_rolling_wfa.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Change rolling window to expanding window
code = code.replace('train_weeks = unique_weeks[max(0, w_idx - MAX_TRAIN_WEEKS) : w_idx]', 'train_weeks = unique_weeks[0 : w_idx]')

# 2. Find the index for the first week of 2026
replacement = """
    # 2026년 첫 번째 주를 찾습니다.
    first_2026_week = next((w for w in unique_weeks if w.startswith("2026-")), None)
    if first_2026_week:
        test_start_idx = unique_weeks.index(first_2026_week)
    else:
        test_start_idx = MIN_TRAIN_WEEKS
"""
code = code.replace('    test_start_idx = MIN_TRAIN_WEEKS  # 104번째 주부터 첫 테스트 시작', replacement)

# 3. Change print messages
code = code.replace('4개년(2022.09 ~ 2026.09)', '2026년 (Expanding Window)')
code = code.replace('직전 104주 (2년 / 504거래일) 고정 윈도우', '최초 데이터부터 직전 주까지 누적 확장(Expanding Window) 학습')

with open('c:\\Users\\chabo\\OneDrive\\바탕 화면\\lumos\\scripts\\temp_run_2026_wfa.py', 'w', encoding='utf-8') as f:
    f.write(code)
print('Temp file created successfully.')
