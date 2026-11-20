"""Current notification contracts. All transports are mocked."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from core.telegram_notifier import TelegramNotifier

class TestLiveDefenseAndBriefing(unittest.TestCase):
    def setUp(self):
        self.notifier = TelegramNotifier()
        self.decision = {
            "direction": "LONG", "dir_gbdt": "LONG", "gating_confidence": .72,
            "is_approved": False, "is_60m_trend_ok": False,
            "threshold_applied": config.GBDT_CONFIDENCE_THRESHOLD,
            "gbdt_probs": {"LONG": .62, "SHORT": .18, "NONE": .20},
            "active_model_id": "TEST_MODEL", "model_consistency_ok": True,
        }

    def test_defense_alert_dispatch(self):
        with patch.object(self.notifier, "_dispatch_message", return_value={"ok": True}) as dispatch:
            result = self.notifier.send_signal_briefing_alert(
                self.decision, "10:15", "23:15", operation_mode="SIMULATION",
                order_allowed=False, order_block_reason="RISK_BLOCK",
                order_block_detail="QQQ_60M_TREND_FILTER", order_executed=False)
        self.assertTrue(result["ok"])
        dispatch.assert_called_once()
        self.assertEqual(dispatch.call_args.kwargs, {"alert_type": "SIGNAL_BRIEFING"})
        msg = dispatch.call_args.args[0]
        for expected in ("\uc8fc\ubb38 \uac00\ub2a5 \uc5ec\ubd80: BLOCKED", "\uc8fc\ubb38 \ucc28\ub2e8 \uc0ac\uc720: RISK_BLOCK",
                         "\ucc28\ub2e8 \uc0c1\uc138: QQQ_60M_TREND_FILTER", "\uc2e4\uc81c \uc8fc\ubb38 \uc2e4\ud589 \uc5ec\ubd80: NO"):
            self.assertIn(expected, msg)

    def test_notifier_briefing_format(self):
        msg = self.notifier.compose_signal_briefing_alert(
            self.decision, "10:15", "23:15", soxl_price=100.0, soxs_price=46.0,
            order_allowed=False, order_executed=False, operation_mode="SIMULATION")
        for expected in ("\ud655\uc2e0\ub3c4: 72.0%", "AI \ud310\ub2e8: LONG", "SOXL $100.00", "SOXS $46.00",
                         f"\uc9c4\uc785 \uc784\uacc4\uac12: {config.GBDT_CONFIDENCE_THRESHOLD * 100:.1f}%",
                         f"{config.BASE_ASSET_LONG}(LONG)", f"{config.BASE_ASSET_SHORT}(SHORT)"):
            self.assertIn(expected, msg)

    def test_veto_dispatch_fractional_confidence_and_escaping(self):
        with patch.object(self.notifier, "_dispatch_message", return_value={"ok": True}) as dispatch:
            self.notifier.send_veto_alert(config.BASE_ASSET_LONG, .72, veto_reason="risk <blocked>", is_simulation=True)
        dispatch.assert_called_once()
        self.assertEqual(dispatch.call_args.kwargs, {"alert_type": "VETO_BLOCK"})
        self.assertIn("72.0%", dispatch.call_args.args[0])
        self.assertIn("risk &lt;blocked&gt;", dispatch.call_args.args[0])
        self.assertNotIn("risk <blocked>", dispatch.call_args.args[0])

if __name__ == "__main__":
    unittest.main()
