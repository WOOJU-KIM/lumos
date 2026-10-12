import re
text = open('core/live_runner.py', encoding='utf-8').read()
matches = re.findall(r'.{0,100}is_appr.{0,100}', text)
with open('scratch/lock_out.txt', 'w', encoding='utf-8') as f:
    for m in matches:
        f.write(m + '\n')
