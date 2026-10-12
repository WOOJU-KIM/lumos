import sys
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
import re

with open('scripts/temp_run_wfa_refactored.py', 'r', encoding='utf-8') as f:
    c = f.read()

# Change the loop to run only for 2026-40
target = "for w_idx in range(len(unique_weeks) - 1, len(unique_weeks)):"
replacement = """
    target_week = '2026-40'
    if target_week in unique_weeks:
        target_idx = unique_weeks.index(target_week)
    else:
        target_idx = len(unique_weeks) - 1
    for w_idx in range(target_idx, target_idx + 1):
"""

c = c.replace(target, replacement)

# Write to a specific temp script
with open('scripts/run_specific_week.py', 'w', encoding='utf-8') as f:
    f.write(c)

print("Created run_specific_week.py")
