# -*- coding: utf-8 -*-
import os

with open('core/kiwoom_broker.py', 'r', encoding='utf-8') as f:
    code = f.read()

target = '''    def get_open_orders(self) -> Dict[str, Any]:'''

replacement = '''    def get_daily_execution_history(self) -> Dict[str, Any]:
        """
        [TR: opt10075 / ust21050] 당일 실시간 체결/매매 내역 조회 (Read-only)
        매매 로직과 100% 분리되어 장 마감 브리핑 등에만 안전하게 사용됨.
        """
        try:
            body = {
                "cano": self.account_no,
                "acnt_prdt_cd": self.account_type,
                "qry_tp": "1"  # 1: 체결내역
            }
            # 에러 발생 시에도 시스템이 멈추지 않도록 raise_on_error=False
            res = self._send_tr_request(endpoint="/api/us/acnt", api_id="opt10075", body_dict=body, raise_on_error=False)
            
            if res.get("return_code") == 0:
                executions = res.get("result_list", [])
                return {
                    "ok": True,
                    "execution_count": len(executions),
                    "executions": executions,
                    "msg": res.get("return_msg", "정상 조회")
                }
            else:
                return {
                    "ok": False,
                    "execution_count": 0,
                    "executions": [],
                    "msg": res.get("return_msg", "조회 실패")
                }
        except Exception as e:
            logger.error(f"체결 내역 조회 중 예외 발생: {e}")
            return {
                "ok": False,
                "execution_count": 0,
                "executions": [],
                "msg": f"에러: {e}"
            }

    def get_open_orders(self) -> Dict[str, Any]:'''

if 'get_daily_execution_history' not in code:
    code = code.replace(target, replacement)
    with open('core/kiwoom_broker.py', 'w', encoding='utf-8') as f:
        f.write(code)
    print("Added get_daily_execution_history!")
