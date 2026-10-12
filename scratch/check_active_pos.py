text = open('core/live_runner.py', encoding='utf-8').read()
idx = text.find('if is_appr and not active_pos :')
print(text[idx-300:idx+300])
