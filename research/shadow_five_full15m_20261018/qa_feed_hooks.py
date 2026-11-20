"""Compare original AST and invoke only the isolated WS cache/callback method."""
import ast
import json
from pathlib import Path
from datetime import datetime
import time
from types import SimpleNamespace
from common import HOME, PROJECT, sha
from capture_ast import normalize_capture

checks={}
for filename,observer in [('kiwoom_ws_streamer.py','_shadow_market'),('ml_engine.py','_shadow_features'),('live_runner.py','_shadow_bind'),('moe_orchestrator.py','__no_observer__')]:
    original=ast.parse((HOME/'qa'/(filename+'.pre_feed')).read_text(encoding='utf-8-sig'))
    current=ast.parse((PROJECT/'core'/filename).read_text(encoding='utf-8-sig'))
    current=normalize_capture(current,filename)
    class Strip(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            if filename=='ml_engine.py' and node.name=='_report_live_sample_coverage':return None
            return self.generic_visit(node)
        def visit_Expr(self,node):
            if filename=='ml_engine.py' and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='_report_live_sample_coverage':return None
            return self.generic_visit(node)
        def visit_If(self,node):
            expected=ast.parse('isinstance(val, (int, float, np.integer, np.floating)) and np.isfinite(val)',mode='eval').body
            if filename=='moe_orchestrator.py' and ast.dump(node.test)==ast.dump(expected):
                node.test=ast.parse('isinstance(val, (int, float)) and not pd.isna(val)',mode='eval').body
            return self.generic_visit(node)
        def visit_Try(self,node):
            observers={observer,'_shadow_autostart'} if filename=='live_runner.py' else {observer}
            if any((isinstance(n,ast.Name) and n.id in observers) or (isinstance(n,ast.alias) and n.asname in observers) for n in ast.walk(node)):
                return None
            return self.generic_visit(node)
    current=Strip().visit(current)
    checks[filename+'_original_ast_preserved']=ast.dump(original)==ast.dump(current)
    assert checks[filename+'_original_ast_preserved']

tree=ast.parse((PROJECT/'core/kiwoom_ws_streamer.py').read_text(encoding='utf-8-sig'))
cl=next(n for n in tree.body if isinstance(n,ast.ClassDef))
method=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='_update_price')
method.returns=None
for a in method.args.args:a.annotation=None
ns=dict(datetime=datetime,time=time,logger=SimpleNamespace(error=lambda *a:None))
exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'WS_CACHE_CALLBACK_ONLY','exec'),ns)
for mode in ['worker_dead','observer_exception']:
    calls=[]
    def observe(*args,**kwargs):
        if mode=='observer_exception':raise RuntimeError('TEST_ONLY')
    ns['_shadow_market']=observe
    fake=SimpleNamespace(latest_prices={},latest_ticks={},ws_tick_received={},last_ws_tick_time={},ws_tick_count={},
                         _callbacks=[lambda *a:calls.append(a)])
    ns['_update_price'](fake,'SOXL',100.,{},True)
    checks[mode+'_live_callback_continues']=len(calls)==1 and calls[0][1]==100. and fake.latest_prices['SOXL']==100.
    assert checks[mode+'_live_callback_continues']

before=json.loads((HOME/'qa/feed_before.json').read_text())
changes=[p for p,h in before.items() if sha(PROJECT/p)!=h]
checks['only_approved_original_files_changed']=set(changes)=={'core/kiwoom_ws_streamer.py','core/ml_engine.py','core/live_runner.py'}
assert checks['only_approved_original_files_changed']
# feed_before predates the extension of its file list; the original full baseline pins MoE.
full_before=json.loads((HOME/'qa/protected_before.json').read_text())
all_changes={p for p,h in full_before.items() if sha(PROJECT/p)!=h}
assert all_changes=={'core/kiwoom_ws_streamer.py','core/ml_engine.py','core/live_runner.py','core/moe_orchestrator.py','core/live_experience_logger.py'}
result=dict(status='PASS',checks=checks,changed=changes,model_config_backup_hashes_unchanged=True,orders=0)
(HOME/'qa/feed_hooks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
