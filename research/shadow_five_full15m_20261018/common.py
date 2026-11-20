"""Candidate-only isolation. Never imports live application modules or credentials."""
import ast
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import platform
import tempfile

HOME = Path(__file__).resolve().parent
PROJECT = HOME.parents[1]
sys.dont_write_bytecode = True
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')
POLICY = json.loads((HOME / 'policy.json').read_text(encoding='utf-8'))
CONFIG = POLICY['production_config']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(relative, value):
    path = HOME / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                               default=lambda x: x.item() if hasattr(x, 'item') else str(x)), encoding='utf-8')


def guard():
    """Deny network, child processes, and every write outside this candidate workspace."""
    platform.uname()  # Cache OS metadata before child-process isolation.
    cache = HOME / 'cache'
    cache.mkdir(exist_ok=True)
    tempfile.tempdir = str(cache)
    os.environ['TORCH_HOME'] = str(cache)
    def denied(*args, **kwargs):
        raise RuntimeError('Candidate isolation: network disabled')
    socket.socket.connect = denied
    socket.create_connection = denied

    def audit(event, args):
        if event == 'open':
            path, mode, flags = args
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if writing and isinstance(path, (str, bytes, os.PathLike)):
                resolved = Path(os.fsdecode(path)).resolve()
                if os.fsdecode(path).lower() == os.devnull.lower():
                    return
                if resolved != HOME and HOME not in resolved.parents:
                    raise RuntimeError('Candidate isolation: outside write rejected')
        if event in ('subprocess.Popen', 'os.system'):
            raise RuntimeError('Candidate isolation: child processes disabled')
    sys.addaudithook(audit)
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)


def protected_changes():
    before = json.loads((HOME / 'qa/protected_before.json').read_text(encoding='utf-8'))
    return [name for name, digest in before.items()
            if not (PROJECT / name).is_file() or sha(PROJECT / name) != digest]


def load_execution_config():
    """Allow only the separately approved fixed-stop config; keep training policy frozen."""
    path = PROJECT / 'config.py'
    if sha(path) == POLICY['production_config_sha256']:
        return dict(CONFIG)
    approval = json.loads((HOME / 'execution_policy.json').read_text(encoding='utf-8'))
    if (approval['schema_version'] != 1 or
            approval['source_policy_sha256'] not in (
                sha(HOME / 'policy.json'),
                hashlib.sha256((HOME / 'policy.json').read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
            ) or
            approval['frozen_config_sha256'] != POLICY['production_config_sha256'] or
            approval['approved_config_sha256'] not in (
                sha(path),
                hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
                hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')).hexdigest(),
            ) or
            approval['overrides'] != {'SL_MAX_PCT': CONFIG['SL_MIN_PCT'], 'TRAILING_TRIGGER_PCT': 0.01}):
        raise RuntimeError('Unapproved execution configuration change')
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    matches = [node for node in tree.body if isinstance(node, ast.Assign) and
               any(isinstance(t, ast.Name) and t.id == 'SL_MAX_PCT' for t in node.targets)]
    if len(matches) != 1 or ast.literal_eval(matches[0].value) != CONFIG['SL_MIN_PCT']:
        raise RuntimeError('Fixed-stop configuration mismatch')
    matches[0].value = ast.Constant(value=CONFIG['SL_MAX_PCT'])
    trail_matches = [node for node in tree.body if isinstance(node, ast.Assign) and
                     any(isinstance(t, ast.Name) and t.id == 'TRAILING_TRIGGER_PCT' for t in node.targets)]
    if len(trail_matches) != 1 or ast.literal_eval(trail_matches[0].value) != approval['overrides']['TRAILING_TRIGGER_PCT']:
        raise RuntimeError('Trailing-trigger configuration mismatch')
    trail_matches[0].value = ast.Constant(value=CONFIG['TRAILING_TRIGGER_PCT'])
    # User-approved live relay settings do not rewrite the five frozen models.
    additional = approval.get('additional_config', {})
    allowed = {'LIVE_EXECUTION_STRATEGY', 'RELAY_INITIAL_ALLOCATION', 'RELAY_CHAIN_ALLOCATION',
               'RELAY_CASH_BUFFER', 'RELAY_OPPOSITE_TRIGGER_PCT', 'RELAY_CHAIN_TRAILING_TRIGGER_PCT',
               'RELAY_QUOTE_MAX_AGE_SECONDS', 'RELAY_LEDGER_REFRESH_SECONDS',
               'RELAY_DECISION_INTERVAL_MINUTES', 'RELAY_STATE_DIRNAME',
               'RELAY_LEGACY_SHADOW_ID', 'RELAY_LEGACY_SHADOW_ENABLED', 'RELAY_EXECUTION_COST_RESERVE',
               'GBDT_FEATURE_GAIN_WEIGHTS', 'CANDLE_READY_RETRY_SECONDS',
               'CANDLE_READY_TIMEOUT_SECONDS', 'ALPACA_BAR_REQUEST_TIMEOUT_SECONDS',
               'RELAY_MAX_CHAIN_COUNT'}
    if set(additional) - allowed:
        raise RuntimeError('Unapproved extra configuration keys')
    kept, found = [], set()
    for node in tree.body:
        names = [t.id for t in node.targets if isinstance(t, ast.Name)] if isinstance(node, ast.Assign) else []
        targeted = set(names) & set(additional)
        if targeted:
            if len(names) != 1 or names[0] in found or ast.literal_eval(node.value) != additional[names[0]]:
                raise RuntimeError('Approved relay configuration mismatch')
            found.add(names[0])
        else:
            kept.append(node)
    if found != set(additional):
        raise RuntimeError('Approved relay settings missing')
    tree.body = kept
    digest = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    if digest != approval['frozen_config_ast_sha256']:
        raise RuntimeError('Configuration changes extend beyond approved fixed stop')
    return dict(CONFIG, **approval['overrides'], **additional)


def verify_source_policy():
    load_execution_config()
    assert CONFIG['MAX_ALLOCATION_RATIO'] == .98
    assert CONFIG['BACKTEST_FEE_SLIPPAGE'] >= .002


def add_torch_dependency():
    """Use an already installed research dependency; no download or installation."""
    import importlib.util
    if importlib.util.find_spec('torch') is None:
        path = Path(os.environ['TEMP']) / 'lumos_doe_20261017_154335/deps'
        if not (path / 'torch').is_dir():
            raise RuntimeError('Existing torch dependency unavailable')
        sys.path.append(str(path))


def model_params():
    return dict(n_estimators=CONFIG['GBDT_N_ESTIMATORS'], max_depth=CONFIG['GBDT_MAX_DEPTH'],
                learning_rate=CONFIG['GBDT_LEARNING_RATE'], subsample=CONFIG['GBDT_FEATURE_FRACTION'],
                colsample_bytree=CONFIG['GBDT_FEATURE_FRACTION'], subsample_freq=1,
                random_state=42, deterministic=True, force_col_wise=True, n_jobs=1, verbosity=-1)


def literal_setting(name):
    if name not in {'BUY_TIMEOUT_SEC','SELL_TIMEOUT_SEC'}:
        raise ValueError('Unapproved setting name')
    tree = ast.parse((PROJECT/'config.py').read_text(encoding='utf-8-sig'))
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
            return float(ast.literal_eval(node.value))
    raise ValueError('Required timing configuration missing')
