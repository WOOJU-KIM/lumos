import re

with open('scripts/temp_run_wfa_10yr.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Let's replace the test_start_idx logic
new_logic = """
    MIN_TRAIN_WEEKS = 104  # 최소 2년
    MAX_TRAIN_WEEKS = 520  # 최대 10년 (Expanding Window)
    
    # 26년 8월 이후 주차 필터링
    target_start_week = '2026-31' # 2026년 8월 첫째 주
    target_start_idx = next((i for i, w in enumerate(unique_weeks) if w >= target_start_week), MIN_TRAIN_WEEKS)
    
    test_start_idx = max(MIN_TRAIN_WEEKS, target_start_idx)
    total_test_weeks = len(unique_weeks) - test_start_idx
    print(f"\\n✅ 테스트 주차: 총 {total_test_weeks}주 ({unique_weeks[test_start_idx]} ~ {unique_weeks[-1]})\\n")
"""

content = re.sub(
    r"MIN_TRAIN_WEEKS = 104.*?print\(f\"\\n✅.*?\\n\"\)",
    new_logic.strip(),
    content,
    flags=re.DOTALL
)

with open('scripts/temp_run_wfa_10yr.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Targeted WFA to start from 2026-31 (August 2026)")
