"""Selection of the three user-approved observers; never imports live trading."""
import json
import logging
import math
import ast
from pathlib import Path

HOME=Path(__file__).resolve().parent


def main_session_start():
    tree=ast.parse((HOME.parents[1]/'config.py').read_text(encoding='utf-8-sig'))
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PHASE_MAIN_START' for t in node.targets):
            return tuple(map(int,ast.literal_eval(node.value).split(':')))
    raise ValueError('Missing configured market entry start')


def comparison_settings():
    value=json.loads((HOME/'comparison_shadows.json').read_text(encoding='utf-8'))
    if (value['schema_version']!=1 or value['enabled'] is not True or
            value['model_source']!='ACTIVE_LIVE_MANIFEST' or
            value['entry_policy']!='MATCH_LIVE_THRESHOLD_AND_NEUTRAL' or
            value['repeat_chain'] is not False or type(value['legacy_tracker_enabled']) is not bool):
        raise ValueError('Unsupported comparison shadow registry')
    modes=[v['mode'] for v in value['strategies'].values()]
    if len(modes)!=3 or set(modes)!={'SINGLE','PRELOSS','AFTER_SL'}:
        raise ValueError('Exactly the three approved execution modes are required')
    if not math.isfinite(value['initial_cash']) or value['initial_cash']<=0:
        raise ValueError('Invalid comparison starting cash')
    return value


def legacy_tracker_enabled():
    """Invalid observer configuration never revives a retired tracker or blocks live."""
    try:
        return comparison_settings()['legacy_tracker_enabled']
    except (OSError,ValueError,TypeError,KeyError):
        logging.getLogger(__name__).warning('Comparison configuration unavailable; legacy tracker stays disabled')
        return False


def comparison_runtime_enabled():
    """Whether the 3 ABS72 15m comparison shadows should run in the live shadow consumer."""
    try:
        return bool(comparison_settings().get('runtime_enabled', True))
    except (OSError,ValueError,TypeError,KeyError):
        return False
