"""Pure WebSocket event decoding and virtual risk clock; no I/O or live imports.

Bid marks are executable-side observations, never fabricated trade ticks. Last
prices are admitted only when an explicit numeric last-price field was captured.
"""
from datetime import datetime
from zoneinfo import ZoneInfo
import math
import struct

TICK_V1 = struct.Struct('<16sdddB')
TICK_V2 = struct.Struct('<16sdddBB8s')
MARKET_V2 = 4
EXPLICIT_LAST = 1
NY = ZoneInfo('America/New_York')


def iso(epoch):
    return datetime.fromtimestamp(epoch, NY).isoformat()


def decode_frame(kind, payload):
    if kind == 1:
        symbol, price, bid, ask, ws = TICK_V1.unpack(payload)
        flags, packet_type, version = 0, 'LEGACY', 1
    elif kind == MARKET_V2:
        symbol, price, bid, ask, ws, flags, packet = TICK_V2.unpack(payload)
        if flags & ~EXPLICIT_LAST:
            raise ValueError('UNKNOWN_MARKET_FLAGS')
        packet_type, version = packet.rstrip(b'\0').decode('ascii'), 2
    else:
        raise ValueError('UNKNOWN_MARKET_PROTOCOL')
    if not ws:
        return None
    if not all(math.isfinite(v) and v >= 0 for v in (price, bid, ask)):
        raise ValueError('INVALID_MARKET_NUMBER')
    if bid > 0 and ask > 0 and ask < bid:
        raise ValueError('CROSSED_QUOTE')
    return dict(symbol=symbol.rstrip(b'\0').decode('ascii'), price=price,
                bid=bid, ask=ask, explicit_last=bool(flags & EXPLICIT_LAST),
                packet_type=packet_type, protocol_version=version)


class MarketEvents:
    def __init__(self, symbols, cadence):
        if cadence <= 0:
            raise ValueError('INVALID_POLL_CADENCE')
        self.symbols = set(symbols)
        self.cadence = cadence
        self.prices = {}
        self.last_poll = None

    def frame(self, decoded, observed_at, now):
        if decoded['symbol'] not in self.symbols:
            return []
        if not 0 <= now - observed_at <= self.cadence:
            raise ValueError('STALE_OR_FUTURE_MARKET_FRAME')
        symbol = decoded['symbol']
        metadata = dict(packet_type=decoded['packet_type'],
                        protocol_version=decoded['protocol_version'],
                        observed_at=iso(observed_at))
        events = []
        bid, ask = decoded['bid'], decoded['ask']
        # Use the observed bid for both risk marks and eventual sell fills.
        # A midpoint or an unverified legacy callback price is never a tick.
        if 0 < bid <= ask:
            events.append(dict(kind='QUOTE', time=iso(now), quote_time=iso(observed_at),
                               symbol=symbol, bid=bid, ask=ask, **metadata))
            price, basis = bid, 'WS_BID'
        elif decoded['explicit_last'] and decoded['price'] > 0:
            price, basis = decoded['price'], 'WS_EXPLICIT_LAST'
        else:
            return [dict(kind='MARKET_REJECTED', time=iso(now), symbol=symbol,
                         reason='NO_VERIFIED_RISK_PRICE', **metadata)]
        self.prices[symbol] = (observed_at, price, basis, metadata)
        events.append(dict(kind='MARK', time=iso(now), symbol=symbol,
                           price=price, price_basis=basis, **metadata))
        return events

    def timer(self, now):
        if self.last_poll is not None and now - self.last_poll < self.cadence:
            return []
        self.last_poll = now
        events = []
        for symbol, (observed_at, price, basis, metadata) in self.prices.items():
            if 0 <= now - observed_at <= self.cadence:
                events.append(dict(kind='POLL', time=iso(now), symbol=symbol,
                                   price=price, price_basis=basis, **metadata))
        # Time/EOD intents must still run when every market feed has stopped.
        events.append(dict(kind='CLOCK', time=iso(now)))
        return events
