import re

with open('scripts/temp_run_wfa_10yr.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove the sandbox override
content = re.sub(
    r"config\.USE_60M_TREND_FILTER = True\nprint\('\*\*\* Sandbox Override: USE_60M_TREND_FILTER = True \*\*\*'\)\n",
    "",
    content
)

# Reset test_start_idx to MIN_TRAIN_WEEKS (104)
content = re.sub(
    r"test_start_idx = next\(\(i for i, w in enumerate\(unique_weeks\) if w >= '2026-01'\), MIN_TRAIN_WEEKS\).*?\n",
    "test_start_idx = MIN_TRAIN_WEEKS\n",
    content
)

with open('scripts/temp_run_wfa_10yr.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Restored full 4.2 year WFA run with default config.")
