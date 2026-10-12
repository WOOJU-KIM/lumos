import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config

with open('scripts/run_specific_week.py', 'r', encoding='utf-8') as f:
    c = f.read()

# Change the target week to 2026-41 (which is Oct 5 to Oct 9)
c = c.replace("target_week = '2026-40'", "target_week = '2026-41'")

with open('scripts/run_specific_week_41.py', 'w', encoding='utf-8') as f:
    f.write(c)

print("Created run_specific_week_41.py")
