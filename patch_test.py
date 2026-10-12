# -*- coding: utf-8 -*-
import os, sys
with open('scripts/run_live_test.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if 'global_runner.broker = FakeBroker' in line:
        lines.insert(i+1, '    print("Created FakeBroker")\n')
    elif 'feeder = MockKiwoomFeeder' in line:
        lines.insert(i+1, '    print("Created MockKiwoomFeeder")\n')
    elif 'global_runner.ws_streamer = feeder' in line:
        lines.insert(i+1, '    print("Assigned ws_streamer")\n')
    elif 'global_runner._wait_for_fill =' in line:
        lines.insert(i+1, '    print("Assigned _wait_for_fill")\n')
    elif 'global_runner.dispatcher.send_telegram_message =' in line:
        lines.insert(i+1, '    print("Assigned send_telegram_message")\n')
with open('scripts/run_live_test.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
