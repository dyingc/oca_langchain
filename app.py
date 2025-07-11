import os
import streamlit as st
import streamlit.components.v1 as components # Added this line
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage
from oca_llm import OCAChatModel
from oca_oauth2_token_manager import OCAOauth2TokenManager

# --- 1. App Configuration ---
st.set_page_config(page_title="OAuth2 Chatbot", page_icon="🤖")
st.title("🤖 OAuth2-Powered Chatbot")
load_dotenv()

# --- 2. Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="你好！我是一个由自定义聊天模型驱动的机器人。有什么可以帮您的吗？")
    ]

# --- 3. Initialize Core Components ---
# 使用 st.session_state 来缓存核心对象，避免在每次交互时都重新创建
if "token_manager" not in st.session_state:
    try:
        st.session_state.token_manager = OCAOauth2TokenManager(dotenv_path=".env")
    except (FileNotFoundError, ValueError) as e:
        st.error(f"初始化认证管理器失败: {e}")
        st.stop()

if "chat_model" not in st.session_state:
    try:
        # 初始化时，模型会使用其内部逻辑设置一个默认模型
        st.session_state.chat_model = OCAChatModel.from_env(
            token_manager=st.session_state.token_manager
        )
    except (FileNotFoundError, ValueError) as e:
        st.error(f"初始化聊天模型失败: {e}")
        st.stop()

# --- 4. Sidebar for Settings ---
with st.sidebar:
    st.header("⚙️ 设置")

    # System Prompt
    custom_system_prompt = st.text_area(
        "系统提示 (System Prompt)",
        value=os.getenv("LLM_SYSTEM_PROMPT", "You are a helpful assistant."),
        height=150
    )

    # Temperature
    custom_temperature = st.slider(
        "模型温度 (Temperature)",
        min_value=0.0, max_value=2.0,
        value=float(os.getenv("LLM_TEMPERATURE", 0.7)),
        step=0.1
    )

    st.divider()

    # Model Selection Area
    st.subheader("模型选择")

    # 从 chat_model 实例中动态获取列表
    available_models = st.session_state.chat_model.available_models
    if not available_models:
        st.warning("无法获取可用模型列表。请检查配置或点击刷新。")

    # 获取当前模型在列表中的索引，如果找不到则默认为0
    try:
        # 确保 st.session_state.chat_model.model 的值在列表中
        if st.session_state.chat_model.model not in available_models:
            # 如果当前模型不在列表中（例如，刷新后列表变了），则使用第一个
            st.session_state.chat_model.model = available_models[0] if available_models else ""
        current_model_index = available_models.index(st.session_state.chat_model.model) if st.session_state.chat_model.model else 0
    except ValueError:
        current_model_index = 0

    selected_model = st.selectbox(
        "选择模型 (Model)",
        options=available_models,
        index=current_model_index,
        # 当列表为空时，禁用选择框
        disabled=not available_models
    )

    # Refresh Button
    if st.button("🔄 刷新模型列表"):
        try:
            st.session_state.chat_model.fetch_available_models()
            # 刷新后，重新运行脚本以更新UI
            st.rerun()
        except Exception as e:
            st.error(f"刷新模型列表失败: {e}")

# --- 5. Update Chat Model with Sidebar Settings ---
# 将侧边栏的设置同步到 chat_model 实例
st.session_state.chat_model.model = selected_model
st.session_state.chat_model.temperature = custom_temperature

# Helper function for copy button
def copy_button(text_to_copy, key):
    """
    Generates a copy button that copies the given text to the clipboard.
    """
    # Escape single quotes and newlines for JavaScript string literal
    escaped_text = text_to_copy.replace("'", "\'").replace("\n", "\\n")
    unique_id = f"copy_button_{key}"
    components.html(
        f"""
        <button id="{unique_id}" onclick="copyTextToClipboard('{escaped_text}')">复制</button>
        <script>
        function copyTextToClipboard(text) {{
            navigator.clipboard.writeText(text).then(function() {{
                console.log('Async: Copying to clipboard was successful!');
            }}, function(err) {{
                console.error('Async: Could not copy text: ', err);
            }});
        }}
        </script>
        <style>
            #{unique_id} {{
                background-color: #4CAF50; /* Green */
                border: none;
                color: white;
                padding: 5px 10px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 12px;
                margin-left: 10px;
                cursor: pointer;
                border-radius: 4px;
            }}
            #{unique_id}:hover {{
                background-color: #45a049;
            }}
        </style>
        """,
        height=30
    )

# --- 6. Display Chat History ---
for i, message in enumerate(st.session_state.chat_history):
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
            copy_button(message.content, f"ai_hist_copy_{i}")
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
            copy_button(message.content, f"human_hist_copy_{i}")

# --- 7. User Input and Chat Logic ---
user_query = st.chat_input("请输入您的问题...")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.write(user_query)
        copy_button(user_query, f"human_new_copy_{len(st.session_state.chat_history) - 1}")

    with st.chat_message("AI"):
        response_placeholder = st.empty()
        full_response = ""

        # 每次调用时都使用最新的配置（包括自定义系统提示）
        messages_for_api = [
            AIMessage(content=custom_system_prompt),
            *[msg for msg in st.session_state.chat_history if isinstance(msg, (HumanMessage, AIMessage))]
        ]

        try:
            # 使用 session_state 中已更新的 chat_model 实例
            stream = st.session_state.chat_model.stream(messages_for_api)
            for chunk in stream:
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
            st.session_state.chat_history.append(AIMessage(content=full_response))
            copy_button(full_response, f"ai_new_copy_{len(st.session_state.chat_history) - 1}") # Add copy button here
        except Exception as e:
            st.error(f"调用API时出错: {e}")