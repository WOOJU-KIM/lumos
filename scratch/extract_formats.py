text_runner = open('core/live_runner.py', encoding='utf-8').read()
text_notifier = open('core/telegram_notifier.py', encoding='utf-8').read()

import re
briefing = re.search(r'briefing_msg = f\"\"\"(.*?)\"\"\"', text_runner, re.DOTALL)
entry = re.search(r'msg = f\"\"\"(.*?)\"\"\"', text_notifier, re.DOTALL)

with open('scratch/formats.txt', 'w', encoding='utf-8') as f:
    f.write('=== Briefing ===\n')
    if briefing: f.write(briefing.group(1) + '\n\n')
    f.write('=== Notifier Messages ===\n')
    
    # Extract all f-strings assigned to 'msg' in telegram_notifier.py
    for m in re.finditer(r'msg = f\"\"\"(.*?)\"\"\"', text_notifier, re.DOTALL):
        f.write(m.group(1) + '\n\n')
