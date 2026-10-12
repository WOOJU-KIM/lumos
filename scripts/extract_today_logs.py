import json

log_file = r'c:\Users\chabo\OneDrive\바탕 화면\lumos\data\system_logs.jsonl'
output_file = r'c:\Users\chabo\OneDrive\바탕 화면\lumos\temp_today_trades.txt'

with open(output_file, 'w', encoding='utf-8') as out:
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()[-100000:]
        for line in lines:
            try:
                data = json.loads(line)
                msg = data.get('message', '')
                dt = data.get('datetime', '')
                level = data.get('level', '')
                keywords = ['손절', '매수', '매도', 'SL', '진입', '청산', 'Trailing', 'Stop']
                if any(k in msg for k in keywords) and ('2026-10-05' in dt or '2026-10-06' in dt):
                    out.write(f'[{dt}] {level}: {msg}\n')
            except Exception as e:
                pass
