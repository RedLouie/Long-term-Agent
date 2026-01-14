import hashlib
import json
import sqlite3
from typing import Optional, List, Dict
import numpy as np
from openai import OpenAI


class Small_Agent:
    def __init__(self, db_path="Red_memory3.db"):
        self.client = OpenAI(api_key=" sk-X9xETgX9p8lZboXMvWQTy7mHKcpmBO1vTJ4sNzsNIB2YohyG",
                             base_url="https://api.chatanywhere.tech/v1")
        self.db_path = db_path
        self.chat_model = "gpt-3.5-turbo-1106"


    def small_chat(self,user_input):

        system_prompt = "你是一个信息总结处理的LLM助手。需要你把用户提问的信息总结成一句不超过10个字的主题"
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7
        )
        ai_response = response.choices[0].message.content
        return ai_response