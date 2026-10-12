import pandas as pd

with open('scripts/run_weekly_rolling_wfa.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('if active_pos is None and time_str < config.PHASE_MAIN_END:', 'if active_pos is None and config.PHASE_MAIN_START <= time_str < config.PHASE_MAIN_END:')
text = text.replace('for w_idx in range(test_start_idx, len(unique_weeks)):', 'for w_idx in range(len(unique_weeks) - 2, len(unique_weeks)):')

debug_print = """
        for i in range(len(test_df)):
            if confidences[i] >= 0.60:
                print(f"[WFA DEBUG] {test_df.iloc[i]['datetime']} | {directions[i]} | Conf: {confidences[i]:.3f}")
"""
text = text.replace("test_df['Confidence'] = confidences", "test_df['Confidence'] = confidences" + debug_print)

with open('scripts/temp_run_wfa_debug.py', 'w', encoding='utf-8') as f:
    f.write(text)
