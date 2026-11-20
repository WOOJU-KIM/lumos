"""Restore only uniquely matched recorded direction scores; no model inference/API.

Default is read-only planning. --apply backs up SQLite before a single transaction.
"""
from pathlib import Path
import argparse
from contextlib import closing
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.shadow_tracker import ShadowTracker


def plan(db, learning):
    ny = ZoneInfo('America/New_York')
    with closing(sqlite3.connect('file:'+Path(learning).as_posix()+'?mode=ro', uri=True)) as conn:
        snapshots = {}
        for (text,) in conn.execute("SELECT payload_json FROM learning_events WHERE event_type='DECISION'"):
            value = json.loads(text); snapshots[value['event_id']] = value
        decisions = []
        for (text,) in conn.execute("SELECT payload_json FROM learning_events WHERE event_type='DECISION_OUTCOME'"):
            value = json.loads(text); snapshot = snapshots.get(value.get('snapshot_id'))
            if snapshot and value.get('decision_time_ny'):
                observed = datetime.fromisoformat(value['decision_time_ny']).astimezone(ny)
                decisions.append((observed, value, snapshot))
    with closing(sqlite3.connect('file:'+Path(db).as_posix()+'?mode=ro', uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(v) for v in conn.execute('SELECT * FROM shadow_logs ORDER BY id')]
    updates, skipped = [], []
    for row in rows:
        if row.get('score_source') not in (None, 'UNAVAILABLE', 'UNAVAILABLE_HISTORICAL'):
            skipped.append(dict(id=row['id'], reason='EXISTING_SCORE_PROVENANCE_PRESERVED')); continue
        if row['gbdt_confidence'] not in (None, 0):
            skipped.append(dict(id=row['id'], reason='EXISTING_NONZERO_SCORE_PRESERVED')); continue
        entered = datetime.fromisoformat(row['virtual_entry_time'])
        entered = entered.replace(tzinfo=ny) if entered.tzinfo is None else entered.astimezone(ny)
        candidates = [v for v in decisions if v[0].date() == entered.date()
                      and abs((v[0]-entered).total_seconds()) <= 2]
        if len(candidates) != 1:
            skipped.append(dict(id=row['id'], reason='NO_UNIQUE_DECISION_WITHIN_2_SECONDS')); continue
        observed, outcome, snapshot = candidates[0]
        probabilities = dict(zip(snapshot['raw_class_order'], snapshot['raw_class_probabilities']))
        direction = 'LONG' if probabilities['LONG'] >= probabilities['SHORT'] else 'SHORT'
        expected_symbol = 'SOXL' if direction == 'LONG' else 'SOXS'
        if row['symbol'] != expected_symbol:
            skipped.append(dict(id=row['id'], reason='DIRECTION_MISMATCH')); continue
        signal = ShadowTracker.entry_signal_from_decision(dict(learning_snapshot=snapshot,
            active_model_sha256=outcome.get('model_sha256')), direction)
        if signal['confidence'] is None:
            skipped.append(dict(id=row['id'], reason='UNVERIFIED_SNAPSHOT')); continue
        signal_id = hashlib.sha256(('BACKFILL|'+str(row['id'])+'|'+snapshot['event_id']).encode()).hexdigest()
        updates.append(dict(id=row['id'], symbol=row['symbol'], entry_time=row['virtual_entry_time'],
            entry_price=row['virtual_entry_price'], old_score=row['gbdt_confidence'],
            signal_id=signal_id, candle_id=None, gbdt_confidence=signal['confidence'],
            learning_snapshot_id=signal['learning_snapshot_id'], model_sha256=signal['model_sha256'],
            score_source='BACKFILLED_RECORDED_DIRECTION_DISPLAYED_SCORE',
            match_delta_seconds=abs((observed-entered).total_seconds())))
    unresolved = {v['id']:v['reason'] for v in skipped if not v['reason'].startswith('EXISTING_')}
    unavailable = [dict(id=row['id'],symbol=row['symbol'],entry_time=row['virtual_entry_time'],
        entry_price=row['virtual_entry_price'],old_score=row['gbdt_confidence'],reason=unresolved[row['id']])
        for row in rows if row['id'] in unresolved]
    return dict(total_rows=len(rows), updates=updates, skipped=skipped, unavailable=unavailable)


def apply(db, result, backup):
    backup = Path(backup)
    if backup.exists(): raise FileExistsError('Refusing to overwrite existing database backup')
    backup.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db)) as source, closing(sqlite3.connect(backup)) as destination:
        source.backup(destination)
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        ShadowTracker.ensure_signal_schema(conn)
        for value in result['updates']:
            changed = conn.execute('''UPDATE shadow_logs SET gbdt_confidence=?,signal_id=?,
                learning_snapshot_id=?,model_sha256=?,score_source=?
                WHERE id=? AND symbol=? AND virtual_entry_time=? AND virtual_entry_price=?
                AND (gbdt_confidence IS NULL OR gbdt_confidence=0)''',
                (value['gbdt_confidence'],value['signal_id'],value['learning_snapshot_id'],
                 value['model_sha256'],value['score_source'],value['id'],value['symbol'],
                 value['entry_time'],value['entry_price']))
            if changed.rowcount != 1: raise RuntimeError('Backfill source changed; transaction rolled back')
            ShadowTracker.write_entry_signal(conn, value)
        for value in result['unavailable']:
            changed = conn.execute('''UPDATE shadow_logs SET gbdt_confidence=NULL,
                score_source='UNAVAILABLE_HISTORICAL'
                WHERE id=? AND symbol=? AND virtual_entry_time=? AND virtual_entry_price=?
                AND (gbdt_confidence IS NULL OR gbdt_confidence=0)''',
                (value['id'],value['symbol'],value['entry_time'],value['entry_price']))
            if changed.rowcount != 1: raise RuntimeError('Unmatched source changed; transaction rolled back')
        assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=Path, default=ROOT/'data/shadow_trades_live.db')
    parser.add_argument('--learning', type=Path, default=ROOT/'data/live_experience.db.learning.db')
    parser.add_argument('--output', type=Path, default=ROOT/'research/shadow_score_fix_20261021/backfill.json')
    parser.add_argument('--backup', type=Path, default=ROOT/'research/shadow_score_fix_20261021/shadow_trades_before.db')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    result = plan(args.db, args.learning)
    if args.apply: apply(args.db, result, args.backup)
    result['applied'] = args.apply
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(total=result['total_rows'],restore=len(result['updates']),
        mark_missing=len(result['unavailable']),skip=len(result['skipped']),applied=args.apply)))
