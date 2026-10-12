import re

with open('AGENTS.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace Rule 1 Learning logic & Add Features
old_rule_1_end = "- **학습 데이터 롤링 원칙**: 매주 직전 2년(104주) 데이터를 기반으로 모델을 학습시켜 최신 상태를 유지한다 (백테스트와 실전 매매 동일)."
new_rule_1_end = """- **최대 10년 누적 확장(Expanding Window) 학습 원칙**: 모델은 과거 데이터를 절대 잊지 않아야 한다. 매주 주말 롤링 학습 시, 기존의 '직전 2년(104주)' 슬라이딩 윈도우 방식을 전면 폐지하고, **데이터베이스에 존재하는 최초 시점부터 타겟 주차 직전까지의 모든 데이터(최소 2년 ~ 최대 10년/520주)를 누적 확장(Expanding Window)하여 학습**시켜야 한다. 이 원칙은 백테스트와 실전 매매에 동일하게 적용된다.
- **135개 핵심 피처(Features) 융합**: AI의 판단은 오직 가격과 보조지표뿐만 아니라 다차원 피처들의 조합으로 이루어진다.
  - **Base Asset (SOXL/SOXS)**: 15분봉 기준 가격, 이동평균, MACD, RSI, ATR, 볼린저밴드 등 기술적 지표
  - **Cross Assets (거시/섹터 역학)**: NVDA(반도체 대장), QQQ(나스닥 지수), VIXY(시장 변동성/공포 지수), IEF(미 국채/금리) 등 선행/상관 지표
  - **Triple Barrier Labels**: 상단(TP), 하단(SL), 기간(Time)을 모두 고려하여 3-Class(Long, Short, Neutral)로 나눈 정밀 타겟 라벨링
  - **Time-based Features**: 요일, 시간대 등 장중 주기성 패턴 (종합 135개 피처)"""

content = content.replace(old_rule_1_end, new_rule_1_end)

# Replace Rule 8.2 Sliding Window logic
old_rule_8_2 = """2. **엄격한 2년(104주) 롤링 학습 원칙:**
   * 타겟 주차(Week $W$)를 테스트할 때, 모델의 학습 데이터는 오직 **직전 104주 (Week $W-104$ ~ $W-1$)**의 15분봉 데이터만 사용해야 한다. 
   * 테스트 대상 주차($W$)의 데이터는 피처 계산, 정규화(Normalization), 타겟 라벨링 등 모델 학습의 어떠한 과정에도 단 1초라도 포함되어서는 안 된다 (완전 미학습 Out-of-Sample)."""

new_rule_8_2 = """2. **최대 10년 누적 확장(Expanding Window) 학습 원칙 (2년 고정 폐지):**
   * 타겟 주차(Week $W$)를 테스트할 때, 과거 2년 치만 잘라 쓰는 Sliding Window를 절대 금지한다.
   * 모델의 학습 데이터는 **데이터베이스에 존재하는 가장 오래된 과거 데이터(Index 0)부터 타겟 주차 직전 (Week $W-1$)까지의 모든 15분봉 데이터(최소 104주 ~ 최대 520주 한도)**를 누적하여 사용해야 한다.
   * 테스트 대상 주차($W$)의 데이터는 피처 계산, 정규화(Normalization), 타겟 라벨링 등 모델 학습의 어떠한 과정에도 단 1초라도 포함되어서는 안 된다 (미래참조 원천 차단, 완전 미학습 Out-of-Sample)."""

content = content.replace(old_rule_8_2, new_rule_8_2)

with open('AGENTS.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("AGENTS.md successfully updated with 10-year Expanding Window and 135 Features documentation.")
