"""Run isolated lifecycle contracts; never start a live Runner or broker."""
import json
from pathlib import Path
import sys
import unittest

HOME=Path(__file__).resolve().parent
ROOT=HOME.parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from test_shadow_lifecycle import TestShadowLifecycle

suite=unittest.defaultTestLoader.loadTestsFromTestCase(TestShadowLifecycle)
result=unittest.TextTestRunner(verbosity=1).run(suite)
report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
            failures=len(result.failures),errors=len(result.errors),
            real_runner_executed=False,actual_worker_launches=0,orders=0,broker_api_calls=0)
(HOME/'qa/autostart_tests.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
if not result.wasSuccessful():raise SystemExit(1)
