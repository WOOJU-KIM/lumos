import re

with open('core/ml_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_loop = '''            assigned = False
            for j in range(i + 1, end_idx):
                h_ret = (highs[j] - p0) / p0
                l_ret = (lows[j] - p0) / p0

                # ?배리??(+3.0% TP vs -2.0% SL)
                hit_long_tp = (h_ret >= take_profit)
                hit_long_sl = (l_ret <= -stop_loss)

                # ??배리??(-3.0% TP for short vs +2.0% SL for short)
                hit_short_tp = (l_ret <= -take_profit)
                hit_short_sl = (h_ret >= stop_loss)

                if hit_long_tp and not hit_long_sl:
                    labels[i] = 1   # TQQQ ??리
                    assigned = True
                    break
                elif hit_short_tp and not hit_short_sl:
                    labels[i] = -1  # SQQQ ???리
                    assigned = True
                    break
                elif (hit_long_sl and hit_short_sl) or (hit_long_tp and hit_long_sl) or (hit_short_tp and hit_short_sl):
                    labels[i] = 0   # ?시 급????이?
                    assigned = True
                    break

            if not assigned:
                labels[i] = 0'''

new_loop = '''            assigned = False
            long_alive = True
            short_alive = True
            
            for j in range(i + 1, end_idx):
                h_ret = (highs[j] - p0) / p0
                l_ret = (lows[j] - p0) / p0

                # 비관적 우선주의: 손절(SL) 라인을 먼저 평가
                if h_ret >= stop_loss:
                    short_alive = False
                if l_ret <= -stop_loss:
                    long_alive = False
                
                # 살아있는 포지션에 한해 익절(TP) 평가
                if long_alive and h_ret >= take_profit:
                    labels[i] = 1
                    assigned = True
                    break
                elif short_alive and l_ret <= -take_profit:
                    labels[i] = -1
                    assigned = True
                    break
                elif not long_alive and not short_alive:
                    labels[i] = 0
                    assigned = True
                    break

            if not assigned:
                labels[i] = 0'''

# Regex to safely replace the loop block
pattern = re.compile(
    r'\s+assigned = False\n'
    r'\s+for j in range\(i \+ 1, end_idx\):\n'
    r'[\s\S]+?'
    r'\s+if not assigned:\n'
    r'\s+labels\[i\] = 0'
)

match = pattern.search(content)
if match:
    indent = match.group(0).split('assigned = False')[0]
    
    repl = f'''{indent}assigned = False
{indent}long_alive = True
{indent}short_alive = True
{indent}for j in range(i + 1, end_idx):
{indent}    h_ret = (highs[j] - p0) / p0
{indent}    l_ret = (lows[j] - p0) / p0

{indent}    if h_ret >= stop_loss:
{indent}        short_alive = False
{indent}    if l_ret <= -stop_loss:
{indent}        long_alive = False
    
{indent}    if long_alive and h_ret >= take_profit:
{indent}        labels[i] = 1
{indent}        assigned = True
{indent}        break
{indent}    elif short_alive and l_ret <= -take_profit:
{indent}        labels[i] = -1
{indent}        assigned = True
{indent}        break
{indent}    elif not long_alive and not short_alive:
{indent}        labels[i] = 0
{indent}        assigned = True
{indent}        break

{indent}if not assigned:
{indent}    labels[i] = 0'''
    content = content.replace(match.group(0), repl)
else:
    print("Pattern not found!")

with open('core/ml_engine.py', 'w', encoding='utf-8') as f:
    f.write(content)
