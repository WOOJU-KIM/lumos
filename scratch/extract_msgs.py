import re

text = open('core/live_runner.py', encoding='utf-8').read()

msgs = []
lines = text.split('\n')
in_msg = False
current_msg = []
for i, line in enumerate(lines):
    if 'send_telegram_message' in line:
        print(f"Line {i}: {line.strip()}")
        # Let's print 5 lines above it too
        for j in range(max(0, i-5), i):
            print(f"  {lines[j].strip()}")
