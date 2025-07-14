import os
import streamlit as st
from dotenv import load_dotenv
import yaml
import re

from langchain_core.messages import AIMessage, HumanMessage
from core.llm import OCAChatModel
from core.oauth2_token_manager import OCAOauth2TokenManager
from ui import conversation_manager as cm
from ui import utils as ui_utils

# --- Load configuration ---
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# --- 1. App Configuration ---
st.set_page_config(page_title="OCA Chat", page_icon="🤖", layout="wide")
st.title("🤖 OCA 大模型聊天机器人")
load_dotenv()

# --- 2. Initialize Core Components & Session State ---
if "token_manager" not in st.session_state:
    try:
        st.session_state.token_manager = OCAOauth2TokenManager(dotenv_path=".env")
    except (FileNotFoundError, ValueError) as e:
        st.error(f"初始化认证管理器失败: {e}")
        st.stop()

if "chat_model" not in st.session_state:
    try:
        st.session_state.chat_model = OCAChatModel.from_env(
            token_manager=st.session_state.token_manager
        )
    except (FileNotFoundError, ValueError) as e:
        st.error(f"初始化聊��模型失败: {e}")
        st.stop()

# Initialize conversation and editing states
cm.initialize_session()
if "show_editor" not in st.session_state:
    st.session_state.show_editor = False
if "edit_index" not in st.session_state:
    st.session_state.edit_index = -1
if "edit_content" not in st.session_state:
    st.session_state.edit_content = ""

# --- 3. Sidebar ---
with st.sidebar:
    st.header("💬 会话管理")
    if st.button("➕ 新建聊天", use_container_width=True):
        cm.new_chat()
        st.rerun()

    st.subheader("历史记录")
    if not st.session_state.get("conversations"):
        st.caption("暂无历史会话")
    else:
        sorted_keys = sorted(
            st.session_state.conversations.keys(),
            key=lambda k: (k != st.session_state.get("current_key"))
        )
        for key in sorted_keys:
            conv_data = st.session_state.conversations[key]
            title = conv_data.get("title", f"聊天-{key[:4]}")
            button_type = "primary" if key == st.session_state.get("current_key") else "secondary"
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                if st.button(f"📜 {title}", key=f"load_{key}", use_container_width=True, type=button_type):
                    cm.load_chat(key)
                    st.rerun()
            with col2:
                st.button("🗑️", key=f"del_{key}", use_container_width=True, on_click=cm.delete_chat, args=(key,))

    st.divider()
    st.header("⚙️ 设置")

    # System Prompt linked to the current conversation state
    # Any change here is automatically saved to session_state
    st.session_state.custom_system_prompt = st.text_area(
        "系统提示 (System Prompt)",
        value=st.session_state.get("custom_system_prompt", config["llm_prompts"]["system_prompt"]),
        height=150,
        key=f"system_prompt_{st.session_state.get('current_key')}"
    )
    custom_temperature = st.slider(
        "模型温度 (Temperature)", 0.0, 2.0, float(os.getenv("LLM_TEMPERATURE", 0.7)), 0.05
    )
    st.divider()
    st.subheader("模型选择")
    available_models = st.session_state.chat_model.available_models
    try:
        current_model_index = available_models.index(st.session_state.chat_model.model)
    except (ValueError, IndexError):
        current_model_index = 0

    selected_model = st.selectbox(
        "选择模型",
        options=available_models,
        index=current_model_index,
        disabled=not available_models
    )

    if st.button("🔄 刷新模型列表"):
        try:
            st.session_state.chat_model.fetch_available_models()
            st.rerun()
        except Exception as e:
            st.error(f"刷新模型列表失败: {e}")

# --- 4. Update Chat Model with Settings ---
st.session_state.chat_model.model = selected_model
st.session_state.chat_model.temperature = custom_temperature

# --- 5. Display Chat History with New Features ---
for i, message in enumerate(st.session_state.get("chat_history", [])):
    role = "AI" if isinstance(message, AIMessage) else "Human"
    with st.chat_message(role):
        # Add copy buttons for AI messages
        if isinstance(message, AIMessage):
            # Use the new function to inject copy buttons into code blocks
            rendered_content = ui_utils.add_copy_to_code_blocks(message.content, f"msg_{i}")
            st.markdown(rendered_content, unsafe_allow_html=True)
            # Add a copy button for the whole response
            ui_utils.render_copy_button(message.content, f"whole_resp_{i}")
        else: # Human message
            st.markdown(message.content)
            # Add "Modify" button for human messages
            if st.button("✏️ 修改", key=f"edit_{i}", help="修改并重新发送此条消息"):
                st.session_state.edit_index = i
                st.session_state.edit_content = message.content
                st.session_state.show_editor = True
                st.rerun()

# --- 6. Centralized Chat Submission Logic ---
def handle_chat_submission():
    """Handles both new user input and resubmissions after editing."""
    # Archive the state before getting the AI response
    cm.archive_current_chat()

    with st.chat_message("AI"):
        response_placeholder = st.empty()
        full_response = ""
        system_prompt = st.session_state.get("custom_system_prompt", "")
        messages_for_api = ([AIMessage(content=system_prompt)] if system_prompt else []) + st.session_state.chat_history

        try:
            stream = st.session_state.chat_model.stream(messages_for_api)
            for chunk in stream:
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)

            st.session_state.chat_history.append(AIMessage(content=full_response))
            cm.archive_current_chat()
            st.rerun()
        except Exception as e:
            st.error(f"调用API时出错: {e}")
            st.rerun()

# --- 7. User Input and Editor UI ---
# The editor UI for modifying a message
if st.session_state.show_editor:
    st.markdown("---")
    st.write("### 📝 修改您的消息")
    new_content = st.text_area("编辑内容:", st.session_state.edit_content, height=150, key="editor_text_area")

    col1, col2, _ = st.columns([1, 1, 5])
    with col1:
        if st.button("🔁 重新发送", use_container_width=True, type="primary"):
            idx = st.session_state.edit_index
            # Update the message content
            st.session_state.chat_history[idx] = HumanMessage(content=new_content)
            # Truncate the history after this message
            st.session_state.chat_history = st.session_state.chat_history[:idx + 1]

            # Hide the editor and trigger a resubmission
            st.session_state.show_editor = False
            handle_chat_submission() # Re-use the submission logic

    with col2:
        if st.button("❌ 取消", use_container_width=True):
            st.session_state.show_editor = False
            st.rerun()

# Standard chat input box, disabled if the editor is active
if user_query := st.chat_input("请输入您的问题...", disabled=st.session_state.show_editor):
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    # Display the new user message immediately
    with st.chat_message("Human"):
        st.markdown(user_query)
    handle_chat_submission()
