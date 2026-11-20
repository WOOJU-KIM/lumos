"""Best-effort pagefile publisher plus an optional background startup helper.

Only the consumer waits. Producer never waits for a consumer/lock/ack. Slow consumers
lose data; consumers record gaps and resynchronize. Import failure is isolated by each caller.
Tick/feature publishing has no broker, disk writer, model or subprocess calls.
Constants belong to this isolated research adapter, never to trading policy.
"""
import mmap
import os
import struct
import threading
import time
import json
import atexit
from pathlib import Path
try:
    from .market_events import TICK_V2, MARKET_V2, EXPLICIT_LAST
except ImportError:
    from market_events import TICK_V2, MARKET_V2, EXPLICIT_LAST

TAG = 'Local\\LumosShadowFiveFull15mV1'
SLOTS = 2048
SLOT = 32768
HEADER = 64
SIZE = HEADER + SLOTS*SLOT
PREFIX = struct.Struct('<QIdB')
TICK = struct.Struct('<16sdddB')
HEARTBEAT_TTL = 2.0
_lock = threading.Lock()
_memory = None
_live_engine = None
_live_streamer = None
_qa_allow = False
_auto_lock = threading.Lock()
_auto_thread = None
_owner_session = None
FEATURE_NAMES = json.loads((Path(__file__).parent/'inputs/schema.json').read_text(encoding='utf-8'))['full_features'][:-1]
_relay_recipe = json.loads((Path(__file__).parent/'relay_directions.json').read_text(encoding='utf-8'))
FEATURE_NAMES += [name for name in _relay_recipe.get('relay_model', {}).get('feature_names', [])
                  if name not in FEATURE_NAMES]


def memory(tag=TAG):
    if os.name != 'nt':
        raise RuntimeError('Windows named pagefile mapping required')
    return mmap.mmap(-1, SIZE, tagname=tag, access=mmap.ACCESS_WRITE)


try:
    _memory = memory()
except Exception:
    pass  # Optional observer must never prevent live startup.


def connected():
    try:
        return _memory is not None and 0 <= time.time()-struct.unpack_from('<d',_memory,8)[0] < HEARTBEAT_TTL
    except Exception:
        return False


def bind(engine, streamer, simulation):
    """Only the real runner's concrete engine/streamer may publish to this feed."""
    global _live_engine, _live_streamer
    if simulation is False and os.environ.get('LUMOS_SHADOW_FEED_ISOLATED')!='1':
        _live_engine=engine
        _live_streamer=streamer


def _ensure_sidecar(owner=None,cancelled=None):
    try:
        from .start_shadow import ensure_started
    except ImportError:
        from start_shadow import ensure_started
    return ensure_started(owner=owner,cancelled=cancelled)


def _record_autostart(state):
    # Called only by the startup helper thread, never the tick/decision publisher.
    state=dict(state,recorded_at=time.time(),real_orders_possible=False)
    (Path(__file__).parent/'qa/feed_autostart.json').write_text(json.dumps(state),encoding='utf-8')


def _autostart_target(session):
    try:
        _record_autostart(_ensure_sidecar(owner=session['owner'],cancelled=session['cancelled']))
    except Exception as error:
        try:_record_autostart(dict(status='START_FAILED_LIVE_CONTINUES',error_type=type(error).__name__))
        except Exception:pass
        import logging
        logging.getLogger(__name__).warning('Shadow auto-start failed; live runner continues')


def autostart(simulation):
    """Request once in a background helper; no spawn, file I/O or readiness wait here."""
    global _auto_thread,_owner_session
    if simulation is not False or os.environ.get('LUMOS_SHADOW_FEED_ISOLATED')=='1':
        return False
    if not _auto_lock.acquire(False):return False
    try:
        if _auto_thread is not None and _auto_thread.is_alive():return False
        if _owner_session is not None and not _owner_session['cancelled'].is_set():return False
        try:
            from .lifecycle import current_owner
        except ImportError:
            from lifecycle import current_owner
        _owner_session=dict(owner=current_owner(),cancelled=threading.Event())
        _auto_thread=threading.Thread(target=_autostart_target,args=(_owner_session,),name='LumosShadowAutoStart',daemon=True)
        _auto_thread.start()
        return True
    finally:
        _auto_lock.release()


def autostop(simulation):
    """Cancel an in-flight start and ask only this Runner's worker to stop.

    No enabled=false write: a future Runner session must start its own worker.
    """
    if simulation is not False or os.environ.get('LUMOS_SHADOW_FEED_ISOLATED')=='1':
        return False
    with _auto_lock:
        session=_owner_session
        if session is None:return False
        session['cancelled'].set()
    try:
        from .lifecycle import request_stop
    except ImportError:
        from lifecycle import request_stop
    request_stop(Path(__file__).parent,session['owner'])
    return True


def _shutdown_at_exit():
    try:autostop(False)
    except Exception:pass  # Worker also watches the actual owner process handle.


atexit.register(_shutdown_at_exit)


def publish(kind, payload):
    if not connected():
        return False
    if not _lock.acquire(False):
        # Counter is best effort under GIL; a gap also becomes visible to consumer.
        try:struct.pack_into('<Q',_memory,16,struct.unpack_from('<Q',_memory,16)[0]+1)
        except Exception:pass
        return False
    try:
        if len(payload)>SLOT-PREFIX.size:
            struct.pack_into('<Q',_memory,16,struct.unpack_from('<Q',_memory,16)[0]+1)
            return False
        seq=struct.unpack_from('<Q',_memory,0)[0]+1
        off=HEADER+(seq%SLOTS)*SLOT
        struct.pack_into('<Q',_memory,off,0)  # invalidate before copy
        _memory[off+PREFIX.size:off+PREFIX.size+len(payload)]=payload
        PREFIX.pack_into(_memory,off,seq,len(payload),time.time(),kind)
        struct.pack_into('<Q',_memory,0,seq)  # publish last
        return True
    except Exception:
        return False
    finally:
        _lock.release()


def market(symbol, price, extra=None, from_ws=False, source=None):
    # REST backup prices remain available to the live runner, never to shadow.
    if not from_ws:
        return
    if not (_qa_allow or source is not None and source is _live_streamer) or not connected():
        return
    try:
        # Explicit numeric whitelist: never forward raw broker messages/accounts.
        values=(extra or {}).get('values',extra or {})
        def number(*keys):
            for key in keys:
                value=values.get(key)
                if value is not None:
                    try:return abs(float(str(value).replace(',','')))
                    except (ValueError,TypeError):pass
            return 0.
        bid=number('51','bid_price','bid');ask=number('41','ask_price','ask')
        last=number('last_price','cur_price','price','10')
        packet_type=str((extra or {}).get('type','')).upper().encode('ascii',errors='ignore')[:8]
        publish(MARKET_V2,TICK_V2.pack(symbol.encode('ascii')[:16],last,bid,ask,
                                    bool(from_ws),EXPLICIT_LAST if last>0 else 0,packet_type))
    except Exception:
        pass


def features(frame, source=None):
    if not (_qa_allow or source is not None and source is _live_engine) or not connected():
        return
    try:
        # Existing frame is read once; no extraction, prediction, DB or serialization.
        # Preserve full numeric columns, exact float64 values and candle availability.
        # Column names are transferred separately as ASCII, bounded by the slot size.
        tail=frame.tail(32)
        names=FEATURE_NAMES
        name_bytes='\0'.join(names).encode('ascii')
        rows=tail[names].to_numpy(dtype='<f8',copy=True)
        times=tail['datetime_dt'].astype('int64').to_numpy(dtype='<i8',copy=True)
        payload=struct.pack('<HHH',len(tail),len(names),len(name_bytes))+name_bytes+times.tobytes()+rows.tobytes()
        publish(2,payload)
    except Exception:
        publish(3,b'FEATURE_CAPTURE_FAILED')


class Reader:
    def __init__(self,tag=TAG):
        import ctypes
        self.kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        self.kernel.CreateMutexW.restype=ctypes.c_void_p
        self.kernel.CreateMutexW.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_wchar_p]
        self.kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        self.kernel.ReleaseMutex.argtypes=[ctypes.c_void_p]
        self.mutex=self.kernel.CreateMutexW(None,True,tag+'_consumer')
        if not self.mutex:raise RuntimeError('Cannot claim shadow consumer mutex')
        if ctypes.get_last_error()==183:
            self.kernel.CloseHandle(self.mutex);raise RuntimeError('Shadow consumer already active')
        self.mem=memory(tag)
        # One consumer owns this mapping. Existing fresh heartbeat rejects duplicates.
        if 0<=time.time()-struct.unpack_from('<d',self.mem,8)[0]<HEARTBEAT_TTL:
            self.mem.close();self.kernel.ReleaseMutex(self.mutex);self.kernel.CloseHandle(self.mutex)
            raise RuntimeError('Shadow consumer already active')
        self.cursor=struct.unpack_from('<Q',self.mem,0)[0]
        self.drops=struct.unpack_from('<Q',self.mem,16)[0]
        # Commit backing pages in the consumer, before advertising readiness.
        for offset in range(HEADER,SIZE,4096):
            self.mem[offset:offset+1]=self.mem[offset:offset+1]
        self.beat()

    def beat(self):
        struct.pack_into('<d',self.mem,8,time.time())

    def read(self):
        self.beat()
        latest=struct.unpack_from('<Q',self.mem,0)[0]
        drops=struct.unpack_from('<Q',self.mem,16)[0]
        if drops!=self.drops or latest-self.cursor>SLOTS:
            raise RuntimeError('FEED_GAP_OR_PRODUCER_DROP')
        if latest<=self.cursor:return None
        seq=self.cursor+1;off=HEADER+(seq%SLOTS)*SLOT
        commit,length,stamp,kind=PREFIX.unpack_from(self.mem,off)
        if commit!=seq or length>SLOT-PREFIX.size:raise RuntimeError('FEED_OVERWRITE')
        payload=bytes(self.mem[off+PREFIX.size:off+PREFIX.size+length])
        if struct.unpack_from('<Q',self.mem,off)[0]!=seq:raise RuntimeError('FEED_OVERWRITE')
        self.cursor=seq
        return seq,stamp,kind,payload

    def close(self):
        struct.pack_into('<d',self.mem,8,0.)
        self.mem.close()
        self.kernel.ReleaseMutex(self.mutex);self.kernel.CloseHandle(self.mutex)

    def resync(self):
        """Consumer only: discard unread history after a ring gap/overwrite."""
        latest=struct.unpack_from('<Q',self.mem,0)[0]
        drops=struct.unpack_from('<Q',self.mem,16)[0]
        lost=max(0,latest-self.cursor)
        self.cursor=latest
        self.drops=drops
        self.beat()
        return lost
