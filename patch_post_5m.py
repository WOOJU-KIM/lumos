import re

with open('core/weekly_tournament.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_post_5m = "post_5m = target_5m_df[target_5m_df['datetime'] > cur_time]"
new_post_5m = "import pandas as pd\n                post_5m = target_5m_df[target_5m_df['datetime'] >= cur_time + pd.Timedelta(minutes=15)]"
content = content.replace(old_post_5m, new_post_5m)

with open('core/weekly_tournament.py', 'w', encoding='utf-8') as f:
    f.write(content)
