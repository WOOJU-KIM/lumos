import os
import re

# Remove from config.py
with open("config.py", "r", encoding="utf-8") as f:
    config_text = f.read()
config_text = re.sub(r'KILL_SWITCH_VIX_SPIKE_PCT\s*=\s*.*?#.*?\n', '', config_text)
config_text = re.sub(r'KILL_SWITCH_TQQQ_DROP_PCT\s*=\s*.*?#.*?\n', '', config_text)
with open("config.py", "w", encoding="utf-8") as f:
    f.write(config_text)

# Remove from __init__.py
with open("core/__init__.py", "r", encoding="utf-8") as f:
    init_text = f.read()
init_text = init_text.replace("from .kill_switch import BlackSwanKillSwitch\n", "")
init_text = init_text.replace('    "BlackSwanKillSwitch",\n', "")
with open("core/__init__.py", "w", encoding="utf-8") as f:
    f.write(init_text)

# Delete file
if os.path.exists("core/kill_switch.py"):
    os.remove("core/kill_switch.py")

print("Kill switch fully purged!")
