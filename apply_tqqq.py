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
    "config.py",
    "core/live_runner.py",
    "core/telegram_notifier.py",
    "core/moe_orchestrator.py",
    "core/weekly_tournament.py",
    "core/state_hub.py",
    "tests/test_system_integrity.py"
]

for file_path in files_to_patch:
    if not os.path.exists(file_path):
        print(f"Skipping {file_path}")
        continue
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Safe replacements
    # We do exact matches to avoid messing up other things, but here simple replace works
    for k, v in replacements.items():
        content = content.replace(k, v)
        
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
        
print("Replacement done.")
