import re
text = open('core/live_runner.py', encoding='utf-8').read()
idx = text.find('open_msg =')
print(text[idx-100:idx+400])
