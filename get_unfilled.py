import json
with open('unfilled_logs.txt', 'w', encoding='utf-8') as f:
    for line in open('data/system_logs.jsonl', encoding='utf-8', errors='ignore'):
        if ('2026-09-21' in line or '2026-09-22' in line) and ('미체결' in line or '취소' in line or 'timeout' in line.lower()):
            try:
                doc = json.loads(line)
                if 'WebSock' not in doc['source']:
                    f.write(f"{doc['datetime']} [{doc['source']}] {doc['message']}\n")
            except:
                pass
