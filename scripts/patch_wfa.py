import re

with open('scripts/temp_run_wfa_10yr.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace ROLLING_WINDOW_WEEKS logic
new_logic = """
    MIN_TRAIN_WEEKS = 104  # 최소 2년의 학습 데이터 확보 후 테스트 시작
    MAX_TRAIN_WEEKS = 520  # 최대 10년의 누적 데이터 학습 (Expanding Window up to 10y)
    if len(unique_weeks) <= MIN_TRAIN_WEEKS:
        raise RuntimeError("데이터 주차가 최소 학습 기간(104주)보다 적습니다.")

    test_start_idx = MIN_TRAIN_WEEKS  # 104번째 주차부터 아웃오브샘플 테스트 시작
    total_test_weeks = len(unique_weeks) - test_start_idx
    print(f"\\n✅ 총 주간 롤링 테스트 주차: 총 {total_test_weeks}주 ({unique_weeks[test_start_idx]} ~ {unique_weeks[-1]})\\n")
"""

content = re.sub(
    r"ROLLING_WINDOW_WEEKS = 104.*?print\(f\".*?\\n\"\)",
    new_logic.strip(),
    content,
    flags=re.DOTALL
)

new_loop_logic = """
    for w_idx in range(test_start_idx, len(unique_weeks)):
        cur_test_week = unique_weeks[w_idx]
        
        # 최대 10년(520주)까지만 데이터를 가져오되, 그보다 적으면 0번 인덱스(처음)부터 가져옴
        train_start_idx = max(0, w_idx - MAX_TRAIN_WEEKS)
        train_weeks = unique_weeks[train_start_idx : w_idx]
        
        # A. 최대 10년 누적 학습 데이터 슬라이싱
"""

content = re.sub(
    r"for w_idx in range\(test_start_idx, len\(unique_weeks\)\):.*?# A. 직전 104주\(2년\) 학습 데이터 슬라이싱",
    new_loop_logic.strip(),
    content,
    flags=re.DOTALL
)

with open('scripts/temp_run_wfa_10yr.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Patched WFA logic")
