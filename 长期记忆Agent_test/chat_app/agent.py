import hashlib
import json
import logging
import sqlite3
from typing import Optional, List, Dict
import numpy as np
from openai import OpenAI


class LongTermMemoryAgent:
    def __init__(self, db_path="Red_memory3.db"):
        self.client = OpenAI(api_key=" sk-X9xETgX9p8lZboXMvWQTy7mHKcpmBO1vTJ4sNzsNIB2YohyG",
                             base_url="https://api.chatanywhere.tech/v1")
        self.db_path = db_path
        # 配置参数
        self.embedding_model = "text-embedding-3-small"
        self.chat_model = "gpt-3.5-turbo-1106"
        self.memory_decay = 0.98  # 每日记忆衰减率
        self.top_k_memories = 5  # 每次检索记忆数量


    def is_username_exist(self, username) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if username:
                cursor.execute("SELECT * FROM chat_app_user WHERE username=?", (username,))
            result = cursor.fetchone()
            return True if result else False

    # ✅新写的
    def get_user_password(self, username):
        with sqlite3.connect(self.db_path) as conn:
            cursor=conn.cursor()
            cursor.execute("SELECT password FROM chat_app_user WHERE username=?",(username,))
            result = cursor.fetchone()
            return result[0] if result else None

    def add_user(self, username: str) -> int:
        """添加用户，返回的是用户的id"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""INSERT OR REPLACE INTO users(username)VALUES(?)""", (username,))
            conn.commit()
            return cursor.lastrowid

    def search_user(self, username: str) -> Optional[int]:
        """根据用户名字查找用户表的user_id，再通过查找user_id去session表里的会话"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM chat_app_user WHERE username=?", (username,))
            result = cursor.fetchone()
            return result[0] if result else None

    def creat_session(self, user_id: int, user_input: str):
        """根据提示的summary还有些别的来创建会话表"""
        # 先是调用chat函数，再是将提问总结成summary，metadata，status，保存到session
        summary_response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[{
                "role": "system", "content": "你是一个信息总结处理的LLM助手。\n"
                                             "需要你把用户提问的信息总结成一句不超过10个字的主题\n"
            }, {"role": "user", "content": f"用户说{user_input}"}
            ]
        )
        summary = summary_response.choices[0].message.content
        metadata_response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[{
                "role": "system", "content": "你是一个信息提取处理的LLM助手。\n"
                                             "需要你对用户的信息提取出一个metadata，即会话的主题以及用户的提示词。"
            }, {"role": "user", "content": f"用户说{user_input}"}
            ]
        )
        metadata = metadata_response.choices[0].message.content
        status = "active"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT  INTO chat_app_chatsession
                (user_id,last_accessed,summary,status,metadata)
                VALUES(?,CURRENT_TIMESTAMP,?,?,?)""", (user_id, summary, status, str(metadata)))
            session_id = cursor.lastrowid  # 获取新生成的 session_id
            conn.commit()
        return session_id

    def export_sessions(self, user_id: int) :
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT summary FROM chat_app_chatsession WHERE user_id = ?", (user_id,))
            sessions = cursor.fetchall()
            if not sessions:
                return None
            return sessions
            # for row in sessions:
            #     d = dict(row)
            #     print(d.values(), type(str))

    def export_session_messages(self, user_input_summary: str) -> Optional[List[Dict]]:
        """感觉还是需要先后存user_input,再是ai_response，否则无法按照时间来陈列出来"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT role,message,last_accessed
                FROM chat_app_chatmessage
                WHERE session_id=(SELECT id FROM chat_app_chatsession WHERE summary=?)""", (user_input_summary,))
            messages = cursor.fetchall()
            return messages
            # for message in messages:
            #     mes = dict(message)
            #     print(mes.values(), type(str))
            # return [dict(row)for row in message] if message else None

    @staticmethod
    def _cosine_similarity(embedding1: bytes, embedding2: bytes) -> float:
        """计算余弦相似度（sqlite自定义函数）"""
        vec1 = np.frombuffer(embedding1, dtype=np.float32)
        vec2 = np.frombuffer(embedding2, dtype=np.float32)
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def _get_memory(self, text: str) -> bytes:
        """获取文本记忆，并将数据序列化"""
        response = self.client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        vector = response.data[0].embedding  # 化为1536维度向量
        return np.array(vector).astype(np.float32).tobytes()  # 精度压缩减少内存并转化成字节数据输出

    def add_memory(self, text: str, role: str, user_id: int, metadata: dict = None):
        # message_id = hashlib.md5(text.encode()).hexdigest()
        message_id = hashlib.md5(text.encode()).hexdigest()
        embedding = self._get_memory(text)
        try:
            with sqlite3.connect(self.db_path) as conn:
                session_id = conn.execute('''
                    SELECT id 
                    FROM chat_app_chatsession
                    WHERE user_id=?
                    ORDER BY last_accessed DESC 
                    LIMIT 1
                    ''', (user_id,)).fetchone()
                if not session_id:
                    self.creat_session(user_id, text)
                    session_id = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
                else:
                    session_id = session_id[0]

                cursor = conn.cursor()

                cursor.execute('''
                    INSERT OR REPLACE INTO chat_app_chatmessage
                    (message_id,session_id,user_id,role,message,last_accessed,embedding,weight,metadata)
                    VALUES (?,?,?,?,?, CURRENT_TIMESTAMP,?,
                    COALESCE((SELECT weight FROM chat_app_chatmessage WHERE message_id =?)*1.1,1.0),?)
                    ''', (message_id, session_id, user_id, role, text, embedding, message_id, str(metadata)))
                conn.commit()
        except Exception as e:
            logging.warning(e)


    def apply_memory_decay(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE chat_app_chatmessage
                SET weight=weight*?
                WHERE DATE (last_accessed) <DATE('now','-1 day')
                ''', (self.memory_decay,))
            conn.execute('''
                DELETE FROM chat_app_chatmessage
                WHERE weight<0.1
                ''')

    def retrieve_messages(self, query: str, session_id: int = None, role: str = None) -> list:
        """
    根据query语义和session_id检索指定会话的历史消息
    :param query: 搜索文本
    :param session_id: 会话ID（需明确传入）
    :param user_id: 用户ID（用于校验权限）
    :param role_filter: 角色过滤（'user','assistant','system'）
    :return: 匹配消息列表，格式：[{"role": "user", "message": "..."}, ...]
    """
        # validate_session_ownership(session_id,)
        self.apply_memory_decay()
        query_embedding = self._get_memory(query)
        with sqlite3.connect(self.db_path) as conn:
            conn.create_function("cosine_similarity", 2, self._cosine_similarity)
            conditions = ["session_id=?"]
            params = [session_id]
            if role:
                conditions.append("role=?")
                params.append(role)
            cursor = conn.execute(f'''
                SELECT role,message,metadata
                FROM chat_app_chatmessage
                WHERE {' AND '.join(conditions)}
                ORDER BY cosine_similarity(embedding,?)*weight DESC 
                LIMIT ?
                    ''', params + [query_embedding, self.top_k_memories])
            messages = cursor.fetchall()
            return [{
                "role": m[0],
                "message": m[1],
                "metadata": m[2] if m[2] else None
            } for m in messages]

    def _get_recent_messages(self, n=10) -> str:
        """提取最近的10条问答"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute('''
                SELECT message,role
                FROM chat_app_chatmessage
                ORDER BY message_id DESC 
                LIMIT ?
                ''', (n,)).fetchall()
        return "\n".join([f"User:{r[0]}\nAI:{r[1]}" for r in reversed(rows)])

    def chat_agent(self, user_input: str, user_id: int) -> str:
        """首先存入user input，然后取出三段记忆：1.最近记忆 2.历史相关记忆 3.(update)metadata（主题+偏好，提示词）
        传给提示词，再把ai的保存到messages"""

        self.add_memory(user_input, "user", user_id)

        with sqlite3.connect(self.db_path) as conn:
            session_id = conn.execute('''
                       SELECT id 
                       FROM chat_app_chatsession
                       WHERE user_id=?
                       ORDER BY last_accessed DESC 
                       LIMIT 1
                       ''', (user_id,)).fetchone()
            if not session_id:
                self.creat_session(user_id, user_input)
                session_id = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
            else:
                session_id = session_id[0]
        memories1 = self.retrieve_messages(user_input, session_id, "user")
        memories2 = self.retrieve_messages(user_input, session_id, "assistant")
        memories = memories1 + memories2
        context = "\n".join([f"[memories]{m}" for m in memories])
        recent_messages = self._get_recent_messages()

        system_prompt = f"""你是一个拥有长期记忆的AI助手，以前的相关记忆以及会话主题提示词：
        {context}

        最近对话记录：
        {recent_messages}

        请自然地进行对话，必要时引用记忆。回复的文本多一点,保持回复不超过最大tokens4096就行"""
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7
        )

        ai_response = response.choices[0].message.content
        self.add_memory(ai_response, "assistant", user_id)

        return ai_response