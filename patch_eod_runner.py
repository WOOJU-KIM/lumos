# -*- coding: utf-8 -*-
import os

with open('core/live_runner.py', 'r', encoding='utf-8') as f:
    code = f.read()

target = '''                        if not already_run:
                            try:
                                from core.data_lake import DailyAutoPipeline'''

replacement = '''                        if not already_run:
                            try:
                                # 증권사 통신: 체결 내역 브리핑 발송 (에러 발생 시에도 무시되도록 try-except 가드)
                                try:
                                    exec_hist = self.broker.get_daily_execution_history()
                                    if exec_hist.get("ok") and exec_hist.get("execution_count", 0) > 0:
                                        msg = f"📊 **[오늘의 실제 매매 체결 내역 (증권사 원장)]**\n\n총 {exec_hist['execution_count']}건의 체결 내역이 확인되었습니다.\n"
                                        for ex in exec_hist["executions"]:
                                            msg += f"• {ex.get('symbol', '종목')} | {ex.get('order_type', '구분')} | {ex.get('quantity', 0)}주 @ \n"
                                        self.dispatcher.send_telegram_message(msg)
                                        system_logger.log("INFO", "ExecutionHistory", f"당일 체결 내역 브리핑 발송 완료 ({exec_hist['execution_count']}건)")
                                except Exception as eh_err:
                                    system_logger.log("ERROR", "ExecutionHistory", f"체결 내역 조회/발송 실패 (무시됨): {eh_err}")
                            except Exception as dummy: pass

                            try:
                                from core.data_lake import DailyAutoPipeline'''

if 'get_daily_execution_history' not in code:
    code = code.replace(target, replacement)
    with open('core/live_runner.py', 'w', encoding='utf-8') as f:
        f.write(code)
    print("Patched EOD runner for execution history!")
