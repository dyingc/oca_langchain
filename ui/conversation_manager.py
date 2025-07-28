import streamlit as st
import uuid
import sys
import os
from langchain.schema import HumanMessage, AIMessage

# Add the project root to the Python path to resolve import issues
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm import OCAChatModel

def initialize_session():
    """Initialize Streamlit session state."""
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    if "current_key" not in st.session_state or st.session_state.current_key not in st.session_state.conversations:
        new_chat()

def new_chat():
    """Create a new chat session."""
    key = uuid.uuid4().hex
    st.session_state.current_key = key
    st.session_state.chat_history = []
    st.session_state.custom_system_prompt = ""
    st.session_state.conversations[key] = {
        "title": "New Chat",
        "system_prompt": "",
        "messages": []
    }

def archive_current_chat():
    """Archive or update the current chat session."""
    key = st.session_state.get("current_key")
    if not key or key not in st.session_state.conversations:
        return

    history = st.session_state.get("chat_history", [])
    system_prompt = st.session_state.get("custom_system_prompt", "")

    current_title = st.session_state.conversations[key].get("title", "New Chat")

    if current_title == "New Chat" and history and "chat_model" in st.session_state:
        title = generate_title(st.session_state.chat_model, history)
    else:
        title = current_title

    st.session_state.conversations[key] = {
        "title": title,
        "system_prompt": system_prompt,
        "messages": history,
    }

def load_chat(key: str):
    """Load the specified chat session."""
    if key in st.session_state.conversations:
        conversation = st.session_state.conversations[key]
        st.session_state.current_key = key
        st.session_state.chat_history = conversation["messages"]
        st.session_state.custom_system_prompt = conversation["system_prompt"]

def delete_chat(key: str):
    """Delete the specified chat session."""
    if key in st.session_state.conversations:
        is_current = st.session_state.current_key == key
        del st.session_state.conversations[key]
        if is_current:
            new_chat()
        st.rerun()

def generate_title(chat_model: OCAChatModel, messages: list) -> str:
    """Use LLM to generate a concise English title for the conversation."""
    if not messages or not chat_model:
        return "Untitled Session"

    history_for_title = "\n".join(
        [f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in messages[:4]]
    )

    prompt = f"""Based on the following conversation, generate a clear and concise English title with a maximum of 5 words. Return only the title, without any explanation or punctuation.

Conversation:
{history_for_title}

Title:"""

    try:
        response = chat_model.invoke(prompt)
        content = response.content if hasattr(response, 'content') else response
        title = content.strip().replace("\"", "").replace("“", "").replace("”", "")
        return title if title else f"Chat-{st.session_state.current_key[:4]}"
    except Exception as e:
        print(f"Error generating title: {e}")
        return f"Chat-{st.session_state.current_key[:4]}"
