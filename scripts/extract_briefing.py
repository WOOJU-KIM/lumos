import codecs
with codecs.open('core/live_runner.py', 'r', 'utf-8') as f:
    content = f.read()
import re
match = re.search(r'(briefing_msg = f"""[^"]+""")', content, re.DOTALL)
if match:
    with open('temp_briefing.txt', 'w', encoding='utf-8') as out:
        out.write(match.group(1))
