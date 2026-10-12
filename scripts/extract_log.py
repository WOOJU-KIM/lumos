import json

with open('data/system_logs.jsonl', 'r', encoding='utf-8') as f, \
     open('data/temp_log_sample.txt', 'w', encoding='utf-8') as out:
    count = 0
    for line in f:
        try:
            d = json.loads(line)
            if 'GBDT' in d.get('message', '') or '최종 확신도' in d.get('message', ''):
                out.write(json.dumps(d, ensure_ascii=False) + '\n')
                count += 1
                if count >= 10: break
        except Exception as e:
            pass
