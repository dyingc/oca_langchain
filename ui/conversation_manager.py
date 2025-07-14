import streamlit as st
import uuid
import sys
import os
from langchain.schema import HumanMessage, AIMessage

# Add the project root to the Python path to resolve import issues
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm import OCAChatModel

def initialize_session():
    """初始化 Streamlit session state。"""
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    if "current_key" not in st.session_state or st.session_state.current_key not in st.session_state.conversations:
        new_chat()

def new_chat():
    """创建一个新的聊天会话。"""
    key = uuid.uuid4().hex
    st.session_state.current_key = key
    st.session_state.chat_history = []
    st.session_state.custom_system_prompt = ""
    st.session_state.conversations[key] = {
        "title": "新聊天",
        "system_prompt": "",
        "messages": []
    }

def archive_current_chat():
    """归档或更新当前会话。"""
    key = st.session_state.get("current_key")
    if not key or key not in st.session_state.conversations:
        return

    history = st.session_state.get("chat_history", [])
    system_prompt = st.session_state.get("custom_system_prompt", "")

    current_title = st.session_state.conversations[key].get("title", "新聊天")

    if current_title == "新聊天" and history and "chat_model" in st.session_state:
        title = generate_title(st.session_state.chat_model, history)
    else:
        title = current_title

    st.session_state.conversations[key] = {
        "title": title,
        "system_prompt": system_prompt,
        "messages": history,
    }

def load_chat(key: str):
    """加载指定的聊天会话。"""
    if key in st.session_state.conversations:
        conversation = st.session_state.conversations[key]
        st.session_state.current_key = key
        st.session_state.chat_history = conversation["messages"]
        st.session_state.custom_system_prompt = conversation["system_prompt"]

def delete_chat(key: str):
    """删除指定的聊天会话。"""
    if key in st.session_state.conversations:
        is_current = st.session_state.current_key == key
        del st.session_state.conversations[key]
        if is_current:
            new_chat()
        st.rerun()

def generate_title(chat_model: OCAChatModel, messages: list) -> str:
    """使用 LLM 为对话生成一个简短的中文标题。"""
    if not messages or not chat_model:
        return "未命名会话"

    history_for_title = "\n".join(
        [f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in messages[:4]]
    )

    prompt = f"""请根据以下对话内容，用中文生成一个5个字以内的、简洁明了的标题。请直接返回标题，不要包含任何多余的解释或标点符号。

对话内容:
{history_for_title}

标题:"""

    try:
        response = chat_model.invoke(prompt)
        content = response.content if hasattr(response, 'content') else response
        title = content.strip().replace("\"", "").replace("“", "").replace("”", "")
        return title if title else f"聊天-{st.session_state.current_key[:4]}"
    except Exception as e:
        print(f"Error generating title: {e}")
        return f"聊天-{st.session_state.current_key[:4]}"
