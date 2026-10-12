import re

with open('core/data_lake.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_code = """        if last_dt == cur_candle_dt:
            # 기존 마지막 캔들 업데이트
            df_copy.loc[last_dt, 'Close'] = live_price
            df_copy.loc[last_dt, 'High'] = max(df_copy.loc[last_dt, 'High'], live_price)
            df_copy.loc[last_dt, 'Low'] = min(df_copy.loc[last_dt, 'Low'], live_price)
        else:
            # 형성 중인 캔들이 아직 수집되지 않은 경우 가짜 캔들을 만들지 않고 이전 종가로 대체하지도 않음.
            # 머신러닝 피처 훼손 방지를 위해 그대로 반환.
            pass"""

new_code = """        if last_dt == cur_candle_dt:
            # 진행 중인 캔들이 DB에 포함되어 있다면 제외 (완성된 캔들로만 피처 생성)
            df_copy = df_copy.drop(last_dt)
        else:
            # 형성 중인 캔들이 아직 DB에 없다면, 그대로 완성된 캔들까지만 반환
            pass"""

content = content.replace(old_code, new_code)

with open('core/data_lake.py', 'w', encoding='utf-8') as f:
    f.write(content)
