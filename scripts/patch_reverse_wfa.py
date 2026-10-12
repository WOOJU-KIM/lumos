import re

with open('scripts/run_reverse_wfa.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace print statements
content = content.replace("주간 롤링 워크포워드(Weekly WFA) 백테스트", "역방향 주간 롤링(Reverse WFA) 백테스트 (미래->과거)")
content = content.replace("backtest_weekly_rolling_wfa", "backtest_reverse_wfa")

# Loop logic
old_loop = """    for w_idx in range(test_start_idx, len(unique_weeks)):
        cur_test_week = unique_weeks[w_idx]
        train_weeks = unique_weeks[max(0, w_idx - MAX_TRAIN_WEEKS) : w_idx]"""

new_loop = """    # 역방향 루프: w_idx를 뒤에서부터 앞으로 이동
    # 가장 마지막 주차에서 MIN_TRAIN_WEEKS(104주)를 뺀 시점부터 과거로 전진
    start_reverse_idx = len(unique_weeks) - config.MIN_TRAIN_WEEKS - 1
    
    for w_idx in range(start_reverse_idx, -1, -1):
        cur_test_week = unique_weeks[w_idx]
        # 학습 데이터: 테스트 주차의 '미래' 주차들 전부 (누적 확장)
        train_weeks = unique_weeks[w_idx + 1 : ]"""

content = content.replace(old_loop, new_loop)

# Progress print
content = content.replace("w_idx - test_start_idx + 1", "start_reverse_idx - w_idx + 1")
content = content.replace("len(unique_weeks) - test_start_idx", "start_reverse_idx + 1")

with open('scripts/run_reverse_wfa.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Reverse WFA script patched.")
