with open("config.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "KILL_SWITCH_VIX_SPIKE_PCT" in line or "KILL_SWITCH_SOXL_DROP_PCT" in line:
        continue
    line = line.replace("SOXL", "TQQQ").replace("SOXS", "SQQQ").replace("SOXX", "QQQ")
    new_lines.append(line)

with open("config.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)
