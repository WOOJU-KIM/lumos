# -*- coding: utf-8 -*-
with open('c:/Users/chabo/OneDrive/바탕 화면/lumos/AGENTS.md', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace("장 마감 10분 전(15:50 NYT, 04:50 KST)", "장 마감 5분 전(15:55 NYT, 04:55 KST)")
text = text.replace("15:30 ~ 15:50 EDT", "15:30 ~ 15:55 EDT")
text = text.replace("15:50 ~ 16:00 EDT", "15:55 ~ 16:00 EDT")

with open('c:/Users/chabo/OneDrive/바탕 화면/lumos/AGENTS.md', 'w', encoding='utf-8') as f:
    f.write(text)
print('Done!')
