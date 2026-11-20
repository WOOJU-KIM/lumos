"""Current absolute-model notification truthfulness; synthetic delivery only."""
import copy,json,sys,unittest
from pathlib import Path
from unittest.mock import MagicMock
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from core.telegram_notifier import TelegramNotifier
from core.ml_engine import MLFeatureEngine
from core.live_runner import KiwoomLiveRunner

class TestCurrentAbsoluteTelegram(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.info=MLFeatureEngine().get_active_model_info()
        cls.manifest=json.loads((ROOT/'models/active_model_manifest.json').read_text(encoding='utf-8'))
    def setUp(self):
        self.n=TelegramNotifier.__new__(TelegramNotifier);self.n._dispatch_message=MagicMock(return_value={'ok':True})
    def decision(self):
        return dict(self.info,active_model_sha256=self.info['sha256'],active_model_schema=self.info['feature_schema'],
            active_model_num_features=self.info['num_features'],direction='LONG',is_approved=True,
            gating_confidence=.7,gbdt_probs=dict(LONG=.7,SHORT=.55,NONE=.5))
    def test_current_model_identity_is_preserved_in_start_open_and_briefing(self):
        d=self.decision();before=copy.deepcopy(d)
        messages=[self.n.compose_startup_alert(self.info),self.n.compose_market_open_alert(self.info,'SIMULATION','SYNTHETIC'),
                  self.n.compose_signal_briefing_alert(d,'10:00','23:00')]
        for message in messages:
            self.assertIn(self.manifest['active_model_id'],message)
            self.assertIn(self.manifest['expected_sha256'][:12],message)
            self.assertIn(self.manifest['expected_schema'],message)
            self.assertNotIn('B_Normalized_72',message)
            self.assertLessEqual(len(message),4096)
        self.assertEqual(d,before)
    def test_missing_metadata_does_not_invent_normalized_or_primary_model(self):
        meta=self.n._extract_model_meta({})
        self.assertEqual(meta['display_status'],'UNVERIFIED')
        self.assertEqual(meta['model_id'],'UNKNOWN');self.assertEqual(meta['schema'],'UNKNOWN')
        self.assertEqual(meta['num_features'],0)
        message=self.n.compose_startup_alert({})
        self.assertNotIn('B_Normalized_72',message);self.assertNotIn('PRIMARY_ACTIVE',message)
    def test_direction_scores_are_not_described_as_probabilities_or_distribution_parity(self):
        message=self.n.compose_signal_briefing_alert(self.decision(),'10:00','23:00')
        self.assertIn('SOXL(LONG) 보정 점수',message)
        self.assertIn('SOXS(SHORT) 보정 점수',message)
        self.assertNotIn('확률:',message);self.assertNotIn('모델 학습 분포와 동일',message)
        self.assertIn('완료 15분봉',message);self.assertIn('WebSocket 호가',message)
    def test_retrain_without_identity_does_not_claim_old_model(self):
        self.n.send_retrain_status_alert(status='MODEL_ACTIVATION_FAILED',error_reason='SYNTHETIC')
        message=self.n._dispatch_message.call_args.args[0]
        self.assertIn('UNKNOWN',message);self.assertNotIn('B_Normalized_72',message)
    def test_fallback_reason_does_not_name_previous_normalized_primary(self):
        runner=KiwoomLiveRunner.__new__(KiwoomLiveRunner);runner.notifier=MagicMock()
        runner._emit_model_state_alerts_from_eval(dict(active_model_is_fallback=True,model_consistency_ok=True))
        kwargs=runner.notifier.send_model_state_alert.call_args.kwargs
        self.assertEqual(kwargs['event_type'],'FALLBACK_TO_CHAMPION')
        self.assertNotIn('B_Normalized_72',kwargs['reason'])

if __name__=='__main__':unittest.main()
