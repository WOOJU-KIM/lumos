from typing import Dict, Any

class BlackSwanKillSwitch:
    """
    [2. Black Swan Kill-Switch (긴급 비상 제동 엔진)]
    장중 VIX 급등(+15% 이상), 지수 급락(-7% 이상) 또는 극단적 이상 징후 발생 시
    모든 포지션을 즉시 전량 청산하고 당일 매매를 원천 중단하는 킬스위치 로직을 수행합니다.
    """
    def __init__(self, vix_spike_limit: float = 15.0, long_drop_limit: float = -7.0, vix_absolute_extreme: float = 35.0):
        self.vix_spike_limit = vix_spike_limit
        self.long_drop_limit = long_drop_limit
        self.vix_absolute_extreme = vix_absolute_extreme

    def evaluate(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """실시간 시세를 스캔하여 블랙스완 킬스위치 발동 여부 검사"""
        vix = market_data.get("vix", {})
        tqqq = market_data.get("assets", {}).get(config.BASE_ASSET_LONG, {})
        
        vix_price = float(vix.get("current_price", 0.0))
        vix_change_pct = float(vix.get("change_pct", 0.0))
        long_change_pct = float(tqqq.get("change_pct", 0.0))
        
        reasons = []
        triggered = False
        
        # 1. VIX 당일 +15% 이상 폭등
        if vix_change_pct >= self.vix_spike_limit:
            triggered = True
            reasons.append(f"VIX 지수 당일 급등 경보 ({vix_change_pct:+.2f}% >= +{self.vix_spike_limit}%)")
            
        # 2. VIX 절대치 35pt 이상 극단적 패닉 장세
        if vix_price >= self.vix_absolute_extreme:
            triggered = True
            reasons.append(f"VIX 절대 수치 극단적 공포 영역 진입 ({vix_price} pt >= {self.vix_absolute_extreme} pt)")
            
        # 3. TQQQ 당일 -7% 이상 급락
        if long_change_pct <= self.long_drop_limit:
            triggered = True
            reasons.append(f"TQQQ 3x 레버리지 당일 폭락 감지 ({long_change_pct:+.2f}% <= {self.long_drop_limit}%)")
            
        if triggered:
            return {
                "triggered": True,
                "status": "EMERGENCY_ACTIVATED",
                "severity": "CRITICAL",
                "title": "🚨 [KILL-SWITCH EMERGENCY ACTIVATED] 긴급 비상 제동 발동",
                "reasons": reasons,
                "action": "EMERGENCY_LIQUIDATE_AND_ABORT",
                "directive": "즉시 전 포지션 전량 청산 및 100% 현금 방어 모드로 전환. 당일 모든 신규 매매 진입을 강제 차단(VETO)합니다.",
                "metrics": {
                    "vix_price": vix_price,
                    "vix_change_pct": vix_change_pct,
                    "long_change_pct": long_change_pct
                }
            }
            
        return {
            "triggered": False,
            "status": "NORMAL_OPERATION",
            "severity": "SAFE",
            "title": "🟢 [KILL-SWITCH STANDBY] 시스템 정상 가동 중",
            "reasons": ["이상 징후 미발생 (VIX 및 가격 변동성 정상 범위 내)"],
            "action": "PROCEED",
            "directive": "정상 퀀트 트레이딩 파이프라인 지속 수행",
            "metrics": {
                "vix_price": vix_price,
                "vix_change_pct": vix_change_pct,
                "long_change_pct": long_change_pct
            }
        }
