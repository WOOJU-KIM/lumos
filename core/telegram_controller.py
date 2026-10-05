import logging
logger = logging.getLogger(__name__)
import time
import json
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DATA_DIR
from core.agentic_brain import AgenticTelegramBrain
from agents.dispatcher_agent import DispatcherAgent

class TelegramController:
    """
    [LLM 기반 대화형 AI 총괄 비서 텔레그램 컨트롤러]
    대표님의 자연어 질문/대화/지시를 실시간 수신하여
    Gemini 3.6 Flash 기반 비서 에이전트가 상황에 맞게 대화형 답변 또는 파라미터 갱신 및 백테스트를 자율 수행
    """
    def __init__(self, bot_token: str = TELEGRAM_BOT_TOKEN, chat_id: str = TELEGRAM_CHAT_ID):
        self.bot_token = bot_token
        self.chat_id = str(chat_id)
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.last_update_id: Optional[int] = None
        self.brain = AgenticTelegramBrain()
        self.dispatcher = DispatcherAgent(self.bot_token, self.chat_id)

    def send_message(self, text: str) -> Dict[str, Any]:
        """텔레그램 메시지 전송"""
        return self.dispatcher.send_telegram_message(text)

    def get_updates(self, timeout: int = 5) -> List[Dict[str, Any]]:
        """신규 텔레그램 메시지 수신 (Long Polling)"""
        url = f"{self.base_url}/getUpdates"
        params = {"timeout": timeout}
        if self.last_update_id is not None:
            params["offset"] = self.last_update_id + 1
        
        try:
            res = requests.get(url, params=params, timeout=timeout + 5)
            data = res.json()
            if data.get("ok"):
                return data.get("result", [])
            return []
        except Exception:
            return []

    def handle_message(self, user_text: str) -> None:
        """대표님의 모든 자연어 메시지를 LLM 대화형 에이전트가 분석 및 즉시 회신"""
        logger.info(f"\n📩 [대표님 메시지 수신] '{user_text}'")
        
        # LLM 대화형 에이전트 추론
        reply_text, is_backtest = self.brain.process_message(user_text)
        
        # 텔레그램 답장 전송
        self.send_message(reply_text)
        action_type_str = "파라미터 갱신 & 백테스트 실행" if is_backtest else "대화형 질문 답변"
        logger.info(f"🚀 >>> [{action_type_str}] 텔레그램 회신 완료 <<< 🚀")

    def listen_loop(self, poll_interval: int = 2, max_iterations: Optional[int] = None):
        """텔레그램 실시간 대화형 수신 루프"""
        logger.info(f"📡 [Telegram Chatbot Agent] AI 총괄 비서 리스너 가동 중 (Chat ID: {self.chat_id})")
        iterations = 0
        
        initial_updates = self.get_updates(timeout=1)
        if initial_updates:
            self.last_update_id = initial_updates[-1]["update_id"]

        while True:
            try:
                updates = self.get_updates(timeout=3)
                for update in updates:
                    self.last_update_id = update["update_id"]
                    msg = update.get("message", {})
                    user_chat_id = str(msg.get("chat", {}).get("id", ""))
                    text = msg.get("text", "")

                    if user_chat_id == self.chat_id and text:
                        self.handle_message(text)

                iterations += 1
                if max_iterations and iterations >= max_iterations:
                    break
                time.sleep(poll_interval)
            except KeyboardInterrupt:
                logger.info("🛑 [Telegram Listener] 리스너 수동 정지")
                break
            except Exception:
                time.sleep(poll_interval)
