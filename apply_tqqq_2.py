import os

replacements = {
    "TQQQ": "TQQQ",
    "SQQQ": "SQQQ",
    "SOXX": "QQQ",
    "tqqq": "tqqq",
    "sqqq": "sqqq",
    "soxx": "qqq",
    "반도체": "나스닥",
    "TQQQ_DROP": "TQQQ_DROP",
}

files_to_patch = [
    "core/kill_switch.py",
    "core/circuit_breaker.py",
    "core/hybrid_broker.py",
    "core/kis_broker.py",
    "core/kiwoom_broker.py",
    "core/portfolio_tracker.py"
]

for file_path in files_to_patch:
    if not os.path.exists(file_path):
        continue
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    for k, v in replacements.items():
        content = content.replace(k, v)
        
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
        
print("Replacement done.")
