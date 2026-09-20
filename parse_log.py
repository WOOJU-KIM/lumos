import json
import codecs

output = []
with codecs.open('data/system_logs.jsonl', 'r', 'utf-8', errors='ignore') as f:
    for line in f:
        try:
            log = json.loads(line)
            dt = log.get('datetime', '')
            if '2026-09-17 22:00' <= dt <= '2026-09-18 05:00':
                if log.get('level') == 'TRADE' or '확신도' in log.get('message', '') or log.get('level') == 'ERROR':
                    output.append(log)
        except Exception as e:
            pass

with codecs.open('parsed_logs.json', 'w', 'utf-8') as out:
    json.dump(output, out, ensure_ascii=False, indent=2)
