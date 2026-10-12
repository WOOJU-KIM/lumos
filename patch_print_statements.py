import re

with open('scripts/run_weekend_data_refresh.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('504 거래일 롤링 윈도우', '최대 10년 누적(Expanding Window)')
content = content.replace('504D 롤링', '최대 10년 누적')
content = content.replace('504 거래일 롤링', '최대 10년 누적')
content = content.replace('최근 504 거래일(Trading Days) 하드코딩', '최대 10년 누적(Expanding Window)')
content = content.replace('(최대 504일 고정 롤링)', '(최대 10년 누적 Expanding)')
content = content.replace('2년 전 과거 데이터 자동 절삭 완료', '모든 과거 데이터 보존 (절삭 없음)')
content = content.replace('-REFRESH-504D', '-REFRESH-EXPANDING')

with open('scripts/run_weekend_data_refresh.py', 'w', encoding='utf-8') as f:
    f.write(content)
