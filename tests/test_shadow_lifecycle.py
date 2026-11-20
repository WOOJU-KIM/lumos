"""Runner/sidecar ownership only. No broker, live Runner or real orders."""
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch, MagicMock

ROOT=Path(__file__).resolve().parents[1]
HOME=ROOT/'research/shadow_five_full15m_20261018'
sys.path.insert(0,str(HOME))
import lifecycle
import start_shadow
import feed_bridge as bridge


class TestShadowLifecycle(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=str(ROOT/'data'))
        self.addCleanup(self.temp.cleanup)
        self.home=Path(self.temp.name)
        (self.home/'qa').mkdir()
        self.owner=lifecycle.current_owner()
        self.control=dict(enabled=True,start_date_ny='2026-10-19',end_date_ny=None,
                          operation='FOLLOW_LIVE_RUNNER',schema_version=1)

    def launch_context(self):
        patches=[patch.object(start_shadow,'__file__',str(self.home/'start_shadow.py')),
                 patch.object(start_shadow,'load_control',return_value=self.control),
                 patch.object(start_shadow,'owner_alive',return_value=True)]
        for p in patches:p.start();self.addCleanup(p.stop)

    def test_exact_process_identity_and_pid_reuse_guard(self):
        self.assertTrue(lifecycle.owner_alive(self.owner))
        self.assertFalse(lifecycle.owner_alive(dict(self.owner,created_filetime=self.owner['created_filetime']+1)))

    def test_stop_token_is_session_scoped(self):
        watch=SimpleNamespace(alive=lambda:True)
        other=dict(self.owner,token='a'*32 if self.owner['token']!='a'*32 else 'b'*32)
        lifecycle.request_stop(self.home,self.owner)
        self.assertEqual(lifecycle.owner_stop_reason(self.home,self.owner,watch),'LIVE_RUNNER_STOP_REQUESTED')
        self.assertIsNone(lifecycle.owner_stop_reason(self.home,other,watch))
        self.assertEqual(lifecycle.owner_stop_reason(self.home,other,SimpleNamespace(alive=lambda:False)),
                         'LIVE_RUNNER_EXITED')

    def test_invalid_owner_cannot_create_stop_path(self):
        with self.assertRaises((ValueError,KeyError,TypeError)):
            lifecycle.stop_path(self.home,dict(self.owner,token='../../stop'))

    def test_follow_mode_does_not_launch_without_runner(self):
        self.launch_context()
        with patch.object(start_shadow.subprocess,'Popen') as spawn:
            self.assertEqual(start_shadow._start_or_reuse()['status'],'WAITING_FOR_LIVE_RUNNER')
        spawn.assert_not_called()

    def test_cancel_before_launch_or_dead_owner_never_spawns(self):
        self.launch_context();cancel=threading.Event();cancel.set()
        with patch.object(start_shadow.subprocess,'Popen') as spawn:
            self.assertEqual(start_shadow._start_or_reuse(self.owner,cancel)['status'],'OWNER_STOPPED_BEFORE_LAUNCH')
            with patch.object(start_shadow,'owner_alive',return_value=False):
                self.assertEqual(start_shadow._start_or_reuse(self.owner)['status'],'OWNER_STOPPED_BEFORE_LAUNCH')
        spawn.assert_not_called()

    def test_spawn_passes_owner_identity_and_records_it(self):
        self.launch_context()
        with patch.object(start_shadow,'connected',side_effect=[False,True]), \
             patch.object(start_shadow.subprocess,'Popen',return_value=SimpleNamespace(pid=12345,poll=lambda:None)) as spawn:
            self.assertEqual(start_shadow._start_or_reuse(self.owner)['status'],'LISTENING')
        command=spawn.call_args.args[0]
        self.assertIn('--owner-pid',command);self.assertIn(str(self.owner['created_filetime']),command)
        self.assertIn(self.owner['token'],command)
        self.assertEqual(json.loads((self.home/'qa/feed_launch.json').read_text())['owner'],self.owner)

    def test_existing_same_owner_reused_other_owner_untouched(self):
        self.launch_context()
        saved=dict(pid=12345,owner=self.owner)
        (self.home/'qa/feed_launch.json').write_text(json.dumps(saved))
        with patch.object(start_shadow,'_same_worker_alive',return_value=True), \
             patch.object(start_shadow,'connected',return_value=True),patch.object(start_shadow.subprocess,'Popen') as spawn:
            self.assertEqual(start_shadow._start_or_reuse(self.owner)['status'],'ALREADY_LISTENING')
            other=dict(self.owner,token='c'*32)
            self.assertEqual(start_shadow._start_or_reuse(other)['status'],'EXISTING_SHADOW_DIFFERENT_OWNER')
        spawn.assert_not_called()

    def test_cancel_during_launch_requests_owned_stop(self):
        self.launch_context();cancel=threading.Event()
        def spawn(*a,**k):
            cancel.set();return SimpleNamespace(pid=12345,poll=lambda:None)
        with patch.object(start_shadow,'connected',return_value=False),patch.object(start_shadow.subprocess,'Popen',side_effect=spawn):
            self.assertEqual(start_shadow._start_or_reuse(self.owner,cancel)['status'],'OWNER_STOPPED_DURING_LAUNCH')
        self.assertTrue(lifecycle.stop_path(self.home,self.owner).exists())

    def test_quick_runner_restart_waits_for_stopping_previous_worker(self):
        self.launch_context()
        previous=dict(self.owner,token='d'*32)
        (self.home/'qa/feed_launch.json').write_text(json.dumps(dict(pid=12345,owner=previous)))
        lifecycle.request_stop(self.home,previous)
        with patch.object(start_shadow,'_same_worker_alive',side_effect=[True,False]), \
             patch.object(start_shadow,'connected',side_effect=[False,True]), \
             patch.object(start_shadow.subprocess,'Popen',return_value=SimpleNamespace(pid=23456,poll=lambda:None)) as spawn:
            self.assertEqual(start_shadow._start_or_reuse(self.owner)['status'],'LISTENING')
        self.assertEqual(spawn.call_count,1)

    def test_background_start_stop_and_new_session(self):
        entered=threading.Event();release=threading.Event();records=[]
        def slow(owner=None,cancelled=None):
            entered.set();release.wait(2)
            return dict(status='CANCELLED' if cancelled.is_set() else 'LISTENING')
        with patch.dict(os.environ),patch.object(bridge,'_owner_session',None),patch.object(bridge,'_auto_thread',None), \
             patch.object(bridge,'_ensure_sidecar',side_effect=slow),patch.object(bridge,'_record_autostart',side_effect=records.append), \
             patch.object(lifecycle,'request_stop') as stop:
            os.environ.pop('LUMOS_SHADOW_FEED_ISOLATED',None)
            try:
                self.assertFalse(bridge.autostart(True))
                self.assertTrue(bridge.autostart(False));self.assertTrue(entered.wait(1))
                token=bridge._owner_session['owner']['token']
                self.assertFalse(bridge.autostart(False))
                self.assertTrue(bridge.autostop(False));stop.assert_called_once()
                release.set();bridge._auto_thread.join(2)
                self.assertEqual(records[-1]['status'],'CANCELLED')
                self.assertTrue(bridge.autostart(False));bridge._auto_thread.join(2)
                self.assertNotEqual(bridge._owner_session['owner']['token'],token)
                bridge.autostop(False)
            finally:release.set();bridge._auto_thread.join(2)

    def test_simulation_and_isolation_do_not_stop_or_start(self):
        with patch.dict(os.environ,{'LUMOS_SHADOW_FEED_ISOLATED':'1'}),patch.object(lifecycle,'request_stop') as stop:
            self.assertFalse(bridge.autostart(False));self.assertFalse(bridge.autostop(False))
            self.assertFalse(bridge.autostop(True))
        stop.assert_not_called()

    def test_runner_finally_stops_on_normal_return_and_errors(self):
        tree=ast.parse((ROOT/'core/live_runner.py').read_text(encoding='utf-8-sig'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='KiwoomLiveRunner')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='run_once_and_start_listener')
        nodes=[n for n in method.body if isinstance(n,ast.Try) and any(isinstance(a,ast.alias) and a.asname in ('_shadow_autostart','_shadow_autostop') for a in ast.walk(n))]
        self.assertEqual(len(nodes),2)
        dry=next(n for n in method.body if isinstance(n,ast.If) and isinstance(n.test,ast.UnaryOp)
                 and isinstance(n.test.operand,ast.Name) and n.test.operand.id=='start_background_threads')
        self.assertLess(dry.lineno,nodes[0].lineno)
        for mode in ('normal','ws_error','listener_interrupt'):
            calls=[];fake=ModuleType('research.shadow_five_full15m_20261018.feed_bridge')
            fake.autostart=lambda simulation:calls.append('start')
            fake.autostop=lambda simulation:calls.append('stop')
            def ws_start():
                if mode=='ws_error':raise RuntimeError('fixture')
            def listen(**kwargs):
                if mode=='listener_interrupt':raise KeyboardInterrupt()
            ns=dict(self=SimpleNamespace(broker=SimpleNamespace(is_simulation=False,broker_name='fixture'),
                    ws_streamer=SimpleNamespace(start=ws_start),controller=SimpleNamespace(listen_loop=listen),
                    _market_execution_loop=lambda:None),logger=MagicMock(),active_model_info={},
                    threading=SimpleNamespace(Thread=lambda **k:SimpleNamespace(start=lambda:None)))
            function=ast.FunctionDef(name='run',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=nodes,decorator_list=[])
            code=compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'START_STOP_TAIL_ONLY','exec')
            with patch.dict(sys.modules,{'research.shadow_five_full15m_20261018.feed_bridge':fake}):
                exec(code,ns)
                try:ns['run']()
                except (RuntimeError,KeyboardInterrupt):pass
            self.assertEqual(calls,['start','stop'],mode)


if __name__=='__main__':unittest.main()
