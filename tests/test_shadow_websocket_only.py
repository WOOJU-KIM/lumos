"""Exercise feed admission without importing a broker or opening the live ring."""
import ast
from pathlib import Path
import struct
import unittest


HOME = Path(__file__).resolve().parents[1] / 'research/shadow_five_full15m_20261018'
TICK = struct.Struct('<16sdddB')
TICK_V2 = struct.Struct('<16sdddBB8s')


def load_function(filename, name, namespace):
    tree = ast.parse((HOME / filename).read_text(encoding='utf-8-sig'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), namespace)
    return namespace[name]


class TestShadowWebsocketOnly(unittest.TestCase):
    def test_publisher_excludes_rest_and_preserves_websocket_quote(self):
        streamer = object()
        sent = []
        ns = dict(TICK=TICK, TICK_V2=TICK_V2, MARKET_V2=4, EXPLICIT_LAST=1,
                  _live_streamer=streamer, _qa_allow=False,
                  connected=lambda: True, publish=lambda *args: sent.append(args))
        market = load_function('feed_bridge.py', 'market', ns)
        market('SOXL', 100., {'provider': 'REST_FALLBACK'}, source=streamer)
        self.assertEqual(sent, [])
        market('SOXL', 101., {'values': {'51': '100.99', '41': '101.01'}},
               from_ws=True, source=streamer)
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0][0], 4)
        self.assertEqual(TICK_V2.unpack(sent[0][1]),
                         (b'SOXL' + b'\0' * 12, 0., 100.99, 101.01, 1, 0, b'\0' * 8))
        market('SOXL', 102., from_ws=True, source=object())
        self.assertEqual(len(sent), 1)

    def test_old_producer_rest_between_ws_ticks_does_not_end_decode(self):
        decode = load_function('live_feed.py', 'decode_market', dict(TICK=TICK))
        payloads = [TICK.pack(b'SOXL', 100., 0., 0., 1),
                    TICK.pack(b'SOXL', 999., 0., 0., 0),
                    TICK.pack(b'SOXS', 42., 41.99, 42.01, 1)]
        rows = [decode(p) for p in payloads]
        self.assertEqual(rows, [('SOXL', 100., 0., 0.), None, ('SOXS', 42., 41.99, 42.01)])

    def test_corrupt_frame_remains_an_error(self):
        decode = load_function('live_feed.py', 'decode_market', dict(TICK=TICK))
        with self.assertRaises(struct.error):
            decode(b'broken')


if __name__ == '__main__':
    unittest.main()
