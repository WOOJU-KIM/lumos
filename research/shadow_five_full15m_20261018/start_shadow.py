"""Hidden sidecar launcher for the CLI and live startup helper. Never starts live trading."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
try:
    from .runtime_control import load_control, should_stop
except ImportError:
    from runtime_control import load_control, should_stop
from datetime import datetime
from zoneinfo import ZoneInfo
try:
    from .feed_bridge import connected
    from .lifecycle import validate_owner, owner_alive, stop_path
except ImportError:
    from feed_bridge import connected
    from lifecycle import validate_owner, owner_alive, stop_path


def _same_worker_alive(saved):
    """PID alone is unsafe after reboot/reuse; also verify process creation time."""
    import ctypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]
    kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
    kernel.GetProcessTimes.argtypes=[ctypes.c_void_p]+[ctypes.POINTER(ctypes.c_ulonglong)]*4
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x1000,False,int(saved['pid']))
    if not handle:return False
    try:
        code=ctypes.c_ulong()
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)) or code.value!=259:return False
        times=[ctypes.c_ulonglong() for _ in range(4)]
        if not kernel.GetProcessTimes(handle,*[ctypes.byref(t) for t in times]):return False
        created=(times[0].value-116444736000000000)/10000000
        return saved.get('worker')=='live_feed.py' and abs(created-float(saved['started_at']))<10
    finally:kernel.CloseHandle(handle)


def _start_or_reuse(owner=None,cancelled=None):
    home=Path(__file__).resolve().parent
    control=load_control()
    if should_stop(datetime.now(ZoneInfo('America/New_York')).date(),control):
        return dict(status='SHADOW_DISABLED_OR_EXPIRED',live_runner_started=False)
    if control.get('operation')=='FOLLOW_LIVE_RUNNER' and owner is None:
        return dict(status='WAITING_FOR_LIVE_RUNNER',live_runner_started=False)
    if owner is not None:owner=validate_owner(owner)
    def stop_requested():
        return (cancelled is not None and cancelled.is_set()) or (owner is not None and not owner_alive(owner))
    if stop_requested():return dict(status='OWNER_STOPPED_BEFORE_LAUNCH',live_runner_started=False)
    pending=home/'qa/feed_launch.json'
    if pending.exists():
        saved=json.loads(pending.read_text())
        if _same_worker_alive(saved):
            if saved.get('owner')!=owner:
                previous=saved.get('owner')
                stopping=False
                if previous is not None:
                    request=stop_path(home,previous)
                    stopping=(not owner_alive(previous) or
                              (request.exists() and json.loads(request.read_text()).get('owner')==previous))
                if not stopping:
                    return dict(status='EXISTING_SHADOW_DIFFERENT_OWNER',pid=saved['pid'],live_runner_started=False)
                # A quick Runner restart can precede the previous worker's next
                # health check. Wait here in the helper, never in live trading.
                deadline=time.monotonic()+50
                while _same_worker_alive(saved):
                    if stop_requested():return dict(status='OWNER_STOPPED_BEFORE_LAUNCH',live_runner_started=False)
                    if time.monotonic()>=deadline:raise RuntimeError('Previous owned shadow did not stop; refusing overlap')
                    time.sleep(.1)
            else:
                if connected():
                    return dict(status='ALREADY_LISTENING',pid=saved['pid'],live_runner_started=False)
                return dict(status='EXISTING_SHADOW_LOADING_OR_RUNNING',pid=saved['pid'])
    if connected():
        return dict(status='EXISTING_SHADOW_IDENTITY_UNVERIFIED',live_runner_started=False)
    if stop_requested():return dict(status='OWNER_STOPPED_BEFORE_LAUNCH',live_runner_started=False)
    command=[sys.executable,'-B',str(home/'live_feed.py')]
    if owner is not None:
        command+=['--owner-pid',str(owner['pid']),'--owner-created',str(owner['created_filetime']),
                  '--owner-token',owner['token']]
    with (home/'qa/feed_stdout.log').open('ab') as out, (home/'qa/feed_stderr.log').open('ab') as err:
        process=subprocess.Popen(command,cwd=str(home),
            stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    pending.write_text(json.dumps(dict(pid=process.pid,started_at=time.time(),worker='live_feed.py',owner=owner)))
    start=time.monotonic()
    while time.monotonic()-start<50:
        if stop_requested():
            if owner is not None:
                try:
                    from .lifecycle import request_stop
                except ImportError:
                    from lifecycle import request_stop
                request_stop(home,owner)
            return dict(status='OWNER_STOPPED_DURING_LAUNCH',pid=process.pid,live_runner_started=False)
        if connected():
            return dict(status='LISTENING',pid=process.pid,live_runner_started=False,
                real_orders_possible=False,market_feed_observed=False)
        if process.poll() is not None:
            raise RuntimeError('Shadow failed to start; see research-only stderr log')
        time.sleep(.1)
    raise RuntimeError('Shadow startup not confirmed within50seconds; do not start a second worker')


def ensure_started(owner=None,cancelled=None):
    """Serialize launchers across processes without waiting; consumer is also singleton."""
    import ctypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateMutexW.restype=ctypes.c_void_p
    kernel.CreateMutexW.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_wchar_p]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    kernel.ReleaseMutex.argtypes=[ctypes.c_void_p]
    handle=kernel.CreateMutexW(None,True,'Local\\LumosShadowFiveFull15mLauncher')
    if not handle:raise RuntimeError('Cannot claim shadow launch mutex')
    if ctypes.get_last_error()==183:
        kernel.CloseHandle(handle)
        return dict(status='ANOTHER_LAUNCHER_ACTIVE',live_runner_started=False)
    try:return _start_or_reuse(owner=owner,cancelled=cancelled)
    finally:
        kernel.ReleaseMutex(handle);kernel.CloseHandle(handle)


if __name__=='__main__':
    print(json.dumps(ensure_started()))
