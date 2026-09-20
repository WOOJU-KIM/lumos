with open('core/live_runner.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('time.sleep(0.05)', 'time.sleep(config.WS_FILL_POLL_INTERVAL)')
content = content.replace('time.sleep(0.5)', 'time.sleep(config.CANCEL_ORDER_WAIT_TIME)')
content = content.replace('sleep_seconds = 3 if mkt[\"is_open\"] else 15', 'sleep_seconds = config.MAIN_LOOP_TICK_OPEN if mkt[\"is_open\"] else config.MAIN_LOOP_TICK_CLOSED')
content = content.replace('sleep_seconds = 3 if mkt[\'is_open\'] else 15', 'sleep_seconds = config.MAIN_LOOP_TICK_OPEN if mkt[\'is_open\'] else config.MAIN_LOOP_TICK_CLOSED')
content = content.replace('time.sleep(1)', 'time.sleep(config.MAIN_LOOP_ERROR_WAIT)')

with open('core/live_runner.py', 'w', encoding='utf-8') as f:
    f.write(content)
