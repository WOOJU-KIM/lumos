import re

with open('scripts/run_weekly_rolling_wfa.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_long = 'calib_conf = min(0.95, max(0.50, 0.50 + (pl - 0.333) * 1.15))'
new_long = 'calib_conf = min(config.CALIB_MAX_CONF, max(config.CALIB_BASE_CONF, config.CALIB_BASE_CONF + (pl - config.CALIB_OFFSET) * config.CALIB_MULTIPLIER))'

old_short = 'calib_conf = min(0.95, max(0.50, 0.50 + (ps - 0.333) * 1.15))'
new_short = 'calib_conf = min(config.CALIB_MAX_CONF, max(config.CALIB_BASE_CONF, config.CALIB_BASE_CONF + (ps - config.CALIB_OFFSET) * config.CALIB_MULTIPLIER))'

content = content.replace(old_long, new_long)
content = content.replace(old_short, new_short)

with open('scripts/run_weekly_rolling_wfa.py', 'w', encoding='utf-8') as f:
    f.write(content)
