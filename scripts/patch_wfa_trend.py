import re

with open('scripts/temp_run_wfa_10yr.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add config override right after imports
if "config.USE_60M_TREND_FILTER = True" not in content:
    content = re.sub(
        r"(import config\n)",
        r"\1config.USE_60M_TREND_FILTER = True\nprint('*** Sandbox Override: USE_60M_TREND_FILTER = True ***')\n",
        content,
        count=1
    )

with open('scripts/temp_run_wfa_10yr.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Applied 60M Trend Filter override to sandbox")
