import json
import re
import pandas as pd
import matplotlib.pyplot as plt

data = []

with open('data/system_logs.jsonl', 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        try:
            d = json.loads(line)
            dt = d.get('datetime', '')
            if not dt.startswith('2026-10'):
                continue
                
            msg = d.get('message', '')
            # Look for percentages like "확신도: 79.9%" or just any "(확신도: XX.X%)" or "15분봉 판단 (확신도: 79.9%)"
            # It might have encoding issues, so we just search for numbers followed by '%'
            # But we should be careful to only get GBDT confidences.
            # "15분봉 판단" has "15" and "%".
            if '15' in msg and '%' in msg:
                # regex to find decimal numbers before '%'
                match = re.search(r'(\d+\.\d+)%', msg)
                if match:
                    conf = float(match.group(1))
                    if 0 <= conf <= 100:  # Valid probability/confidence
                        data.append({'datetime': dt, 'confidence': conf})
        except:
            pass

df = pd.DataFrame(data)
if not df.empty:
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').drop_duplicates('datetime')
    df.to_csv('data/oct_gbdt_conf.csv', index=False)
    print(f"Extracted {len(df)} points for October.")
else:
    print("No data extracted.")
