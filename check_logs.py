import json
with open('data/system_logs.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if 'GBDT' in line or '15' in line:
            print(line)
            if i > 100: break
