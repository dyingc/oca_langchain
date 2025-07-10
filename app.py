import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage
from custom_llm import CustomOauthChatModel
from oauth2_token_manager import Oauth2TokenManager

# --- 1. App Configuration ---
st.set_page_config(page_title="OAuth2 Chatbot", page_icon="🤖")
st.title("🤖 OAuth2-Powered Chatbot")
load_dotenv()

# --- 2. Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="你好！我是一个由自定义聊天模型驱动的机器人。有什么可以帮您的吗？")
    ]

# --- 3. Sidebar for Settings ---
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
    
    # Model Selection
    available_models = os.getenv("LLM_AVAILABLE_MODELS", "").split(',')
    selected_model = st.selectbox(
        "选择模型 (Model)",
        options=available_models,
        index=0 # Default to the first model
    )

# --- 4. Chat Model Initialization ---
# @st.cache_resource cannot hash the token_manager, so we initialize it directly
# This is acceptable for this app's scope.
try:
    token_manager = Oauth2TokenManager(dotenv_path=".env")
    chat_model = CustomOauthChatModel(
        api_url=os.getenv("LLM_API_URL"),
        model=selected_model,
        temperature=custom_temperature,
        token_manager=token_manager
    )
    # We need to update the system prompt in the model if it changes
    # This part is tricky with LangChain's immutable models, so we'll handle it in the payload.
except (FileNotFoundError, ValueError) as e:
    st.error(f"初始化模型失败: {e}")
    st.stop()

# --- 5. Display Chat History ---
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# --- 6. User Input and Chat Logic ---
user_query = st.chat_input("请输入您的问题...")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.write(user_query)

    with st.chat_message("AI"):
        # Use a placeholder for the streaming response
        response_placeholder = st.empty()
        full_response = ""
        
        # Construct messages for the model, including the custom system prompt
        # This overrides the model's default system prompt for this specific call
        messages_for_api = [
            AIMessage(content=custom_system_prompt), # Treat it as a system message for the API
            *[msg for msg in st.session_state.chat_history if isinstance(msg, (HumanMessage, AIMessage))]
        ]

        try:
            stream = chat_model.stream(messages_for_api)
            for chunk in stream:
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
            st.session_state.chat_history.append(AIMessage(content=full_response))
        except Exception as e:
            st.error(f"调用API时出错: {e}")
