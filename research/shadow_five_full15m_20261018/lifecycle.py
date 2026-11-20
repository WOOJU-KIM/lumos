"""Runner ownership by Windows process handle + creation time + session token.

No broker or live application imports. Holding a handle identifies one process
even if its PID is subsequently reused. Never terminates another process.
"""
import ctypes
import json
import os
from pathlib import Path
from uuid import uuid4, UUID


def kernel():
    dll=ctypes.WinDLL('kernel32',use_last_error=True)
    dll.OpenProcess.restype=ctypes.c_void_p
    dll.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]
    dll.GetProcessTimes.argtypes=[ctypes.c_void_p]+[ctypes.POINTER(ctypes.c_ulonglong)]*4
    dll.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
    dll.CloseHandle.argtypes=[ctypes.c_void_p]
    return dll


def validate_owner(owner):
    if (not isinstance(owner,dict) or type(owner.get('pid')) is not int or owner['pid']<=0
            or type(owner.get('created_filetime')) is not int or owner['created_filetime']<=0
            or UUID(owner['token']).hex!=owner['token']):
        raise ValueError('Invalid shadow owner identity')
    return dict(pid=owner['pid'],created_filetime=owner['created_filetime'],token=owner['token'])


class ProcessWatch:
    def __init__(self,pid,created_filetime=None):
        self.dll=kernel()
        self.handle=self.dll.OpenProcess(0x100000|0x1000,False,int(pid))
        self.created_filetime=None
        if self.handle:
            times=[ctypes.c_ulonglong() for _ in range(4)]
            if self.dll.GetProcessTimes(self.handle,*[ctypes.byref(v) for v in times]):
                self.created_filetime=times[0].value
            if self.created_filetime is None or (created_filetime is not None and self.created_filetime!=created_filetime):
                self.close()

    def alive(self):
        return bool(self.handle and self.dll.WaitForSingleObject(self.handle,0)==258)

    def close(self):
        if self.handle:
            self.dll.CloseHandle(self.handle)
            self.handle=None


def current_owner():
    watch=ProcessWatch(os.getpid())
    try:
        if not watch.alive():raise RuntimeError('Cannot establish live runner identity')
        return dict(pid=os.getpid(),created_filetime=watch.created_filetime,token=uuid4().hex)
    finally:watch.close()


def owner_alive(owner):
    owner=validate_owner(owner)
    watch=ProcessWatch(owner['pid'],owner['created_filetime'])
    try:return watch.alive()
    finally:watch.close()


def stop_path(home,owner):
    owner=validate_owner(owner)
    return Path(home)/'qa'/('feed_stop_owner_'+owner['token']+'.json')


def request_stop(home,owner):
    owner=validate_owner(owner)
    path=stop_path(home,owner)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(owner=owner,reason='LIVE_RUNNER_STOPPED')),encoding='utf-8')
    os.replace(temporary,path)


def owner_stop_reason(home,owner,watch):
    path=stop_path(home,owner)
    if path.exists() and json.loads(path.read_text(encoding='utf-8')).get('owner')==owner:
        return 'LIVE_RUNNER_STOP_REQUESTED'
    if not watch.alive():return 'LIVE_RUNNER_EXITED'
    return None
