import re

with open("core/ml_engine.py", "r", encoding="utf-8") as f:
    content = f.read()

pattern = r'if not f_json or str\(f_json\)\.strip\(\) in \(\"\", \"\{\}\", \"nan\"\) or gbdt_conf < 0\.01:'
replacement = r'''act_entry_px = float(row.get("actual_entry_price", 1.0))
                if act_entry_px <= 0.01 or not f_json or str(f_json).strip() in ("", "{}", "nan") or gbdt_conf < 0.01:'''

new_content = re.sub(pattern, replacement, content)

if new_content != content:
    with open("core/ml_engine.py", "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Fixed core/ml_engine.py")
else:
    print("Pattern not found in core/ml_engine.py")
