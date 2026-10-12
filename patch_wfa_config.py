import re

with open('scripts/run_weekly_rolling_wfa.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the wrong config values
content = content.replace('config.CALIB_MAX_CONF', '0.95')
content = content.replace('config.CALIB_BASE_CONF', 'config.CALIB_BASE')
content = content.replace('config.CALIB_OFFSET', 'config.CALIB_PIVOT')
content = content.replace('config.CALIB_MULTIPLIER', 'config.CALIB_SCALE')

with open('scripts/run_weekly_rolling_wfa.py', 'w', encoding='utf-8') as f:
    f.write(content)
