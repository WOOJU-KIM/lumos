"""Shadow execution dates only; frozen training policy/artifacts stay untouched."""
from datetime import date
import json
from pathlib import Path

CONTROL_PATH = Path(__file__).parent / 'runtime_control.json'


def load_control(path=CONTROL_PATH):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if (not isinstance(value, dict) or value.get('schema_version') != 1
            or type(value.get('enabled')) is not bool):
        raise ValueError('Invalid shadow runtime control')
    start = date.fromisoformat(value['start_date_ny'])
    end_text = value['end_date_ny']
    end = date.fromisoformat(end_text) if end_text is not None else None
    if end is not None and end < start:
        raise ValueError('Shadow runtime end precedes start')
    return value


def date_allowed(day, control):
    return bool(control['enabled'] and str(day) >= control['start_date_ny']
                and (control['end_date_ny'] is None or str(day) <= control['end_date_ny']))


def should_stop(day, control):
    return not control['enabled'] or (control['end_date_ny'] is not None
                                     and str(day) > control['end_date_ny'])
