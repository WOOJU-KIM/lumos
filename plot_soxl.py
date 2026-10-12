import re
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import pandas as pd
import base64
import sys
import shutil

# Set up Korean font for Windows
font_path = "C:/Windows/Fonts/malgun.ttf"
try:
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font)
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

def parse_log(filepath):
    months = []
    balances = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Split by lines and find the start of table 3
    lines = content.split('\n')
    in_table_3 = False
    for line in lines:
        if '[3.' in line:
            in_table_3 = True
            continue
        if in_table_3:
            if '결과' in line and '저장' in line: # End of table
                break
            parts = line.split('|')
            if len(parts) >= 3 and '-' in parts[0]:
                month_str = parts[0].strip()
                if re.match(r'^\d{4}-\d{2}$', month_str):
                    bal_str = parts[2]
                    bal_str = re.sub(r'[^\d-]', '', bal_str)
                    try:
                        bal = int(bal_str)
                        months.append(month_str)
                        balances.append(bal)
                    except:
                        pass
    return months, balances

months_off, bal_off = parse_log(r'C:\Users\chabo\.gemini\antigravity\brain\a4ba377f-2239-4e2c-9092-434e9a80c2fd\.system_generated\tasks\task-389.log')
months_on, bal_on = parse_log(r'C:\Users\chabo\.gemini\antigravity\brain\a4ba377f-2239-4e2c-9092-434e9a80c2fd\.system_generated\tasks\task-476.log')

print(f"OFF points: {len(months_off)}, ON points: {len(months_on)}")

all_months = sorted(list(set(months_off + months_on)))
if not all_months:
    print("NO DATA FOUND!")
    sys.exit(1)

df = pd.DataFrame(index=all_months)
df['OFF'] = pd.Series(dict(zip(months_off, bal_off)))
df['ON'] = pd.Series(dict(zip(months_on, bal_on)))

df.ffill(inplace=True)
df.fillna(10000000, inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['OFF'] / 10000, label='60분봉 OFF (공격형 스캘퍼, 최종 +280.5%)', color='red', linewidth=2, marker='o', markersize=4)
plt.plot(df.index, df['ON'] / 10000, label='60분봉 ON (하락장 방어형, 최종 +78.7%)', color='blue', linewidth=2, marker='s', markersize=4)

plt.title('SOXL/SOXS 백테스트 자산 추이 비교 (1천만원 시작)', fontsize=16, fontweight='bold')
plt.xlabel('연월 (Year-Month)', fontsize=12)
plt.ylabel('잔고 (만원)', fontsize=12)

plt.xticks(df.index[::4], rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()

out_path = r'C:\Users\chabo\.gemini\antigravity\brain\a4ba377f-2239-4e2c-9092-434e9a80c2fd\soxl_comparison.png'
plt.savefig(out_path, dpi=150)
print("Chart regenerated.")

with open(out_path, "rb") as f:
    b64_str = base64.b64encode(f.read()).decode('utf-8')
data_uri = f"data:image/png;base64,{b64_str}"

md_path = r"C:\Users\chabo\.gemini\antigravity\brain\a4ba377f-2239-4e2c-9092-434e9a80c2fd\soxl_comparison_report.md"
with open(md_path, "r", encoding="utf-8") as f:
    content = f.read()

content = re.sub(r'!\[SOXL 60분봉 ON vs OFF 비교차트\]\(data:image/png;base64,[A-Za-z0-9+/=]+\)', f'![SOXL 60분봉 ON vs OFF 비교차트]({data_uri})', content)

with open(md_path, "w", encoding="utf-8") as f:
    f.write(content)
print("Markdown updated.")

shutil.copy(out_path, r"C:\Users\chabo\OneDrive\바탕 화면\SOXL_비교차트.png")
