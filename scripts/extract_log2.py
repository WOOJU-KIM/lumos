import json
import re

with open('data/system_logs.jsonl', 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if '15분봉' in line or '확신도' in line or 'gbdt' in line.lower() or 'confidence' in line.lower() or '15m' in line.lower():
            try:
                d = json.loads(line)
                msg = d.get('message', '')
                if '15분봉' in msg or '확신도' in msg or '15m' in msg.lower():
                    print(msg.encode('utf-8').decode('cp949', errors='ignore'))
                    break
            except:
                pass
