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
st.title("🤖 OCA Large Model Chatbot")
load_dotenv()

# --- 2. Initialize Core Components & Session State ---
if "token_manager" not in st.session_state:
    try:
        st.session_state.token_manager = OCAOauth2TokenManager(dotenv_path=".env")
    except (FileNotFoundError, ValueError) as e:
        st.error(f"Failed to initialize authentication manager: {e}")
        st.stop()

if "chat_model" not in st.session_state:
    try:
        st.session_state.chat_model = OCAChatModel.from_env(
            token_manager=st.session_state.token_manager
        )
    except (FileNotFoundError, ValueError) as e:
        st.error(f"Failed to initialize chat model: {e}")
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
    st.header("💬 Conversation Management")
    if st.button("➕ New Chat", use_container_width=True):
        cm.new_chat()
        st.rerun()

    st.subheader("History")
    if not st.session_state.get("conversations"):
        st.caption("No previous conversations")
    else:
        sorted_keys = sorted(
            st.session_state.conversations.keys(),
            key=lambda k: (k != st.session_state.get("current_key"))
        )
        for key in sorted_keys:
            conv_data = st.session_state.conversations[key]
            title = conv_data.get("title", f"Chat-{key[:4]}")
            button_type = "primary" if key == st.session_state.get("current_key") else "secondary"
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                if st.button(f"📜 {title}", key=f"load_{key}", use_container_width=True, type=button_type):
                    cm.load_chat(key)
                    st.rerun()
            with col2:
                st.button("🗑️", key=f"del_{key}", use_container_width=True, on_click=cm.delete_chat, args=(key,))

    st.divider()
    st.header("⚙️ Settings")

    # System Prompt linked to the current conversation state
    # Any change here is automatically saved to session_state
    st.session_state.custom_system_prompt = st.text_area(
        "System Prompt",
        value=st.session_state.get("custom_system_prompt", config["llm_prompts"]["system_prompt"]),
        height=150,
        key=f"system_prompt_{st.session_state.get('current_key')}"
    )
    custom_temperature = st.slider(
        "Model Temperature", 0.0, 2.0, float(os.getenv("LLM_TEMPERATURE", 0.7)), 0.05
    )
    st.divider()
    st.subheader("Model Selection")
    available_models = st.session_state.chat_model.available_models
    try:
        current_model_index = available_models.index(st.session_state.chat_model.model)
    except (ValueError, IndexError):
        current_model_index = 0

    selected_model = st.selectbox(
        "Select Model",
        options=available_models,
        index=current_model_index,
        disabled=not available_models
    )

    if st.button("🔄 Refresh Model List"):
        try:
            st.session_state.chat_model.fetch_available_models()
            st.rerun()
        except Exception as e:
            st.error(f"Failed to refresh model list: {e}")

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
            if st.button("✏️ Edit", key=f"edit_{i}", help="Edit and resend this message"):
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
            st.error(f"Error calling API: {e}")
            st.rerun()

# --- 7. User Input and Editor UI ---
# The editor UI for modifying a message
if st.session_state.show_editor:
    st.markdown("---")
    st.write("### 📝 Edit Your Message")
    new_content = st.text_area("Edit Content:", st.session_state.edit_content, height=150, key="editor_text_area")

    col1, col2, _ = st.columns([1, 1, 5])
    with col1:
        if st.button("🔁 Resend", use_container_width=True, type="primary"):
            idx = st.session_state.edit_index
            # Update the message content
            st.session_state.chat_history[idx] = HumanMessage(content=new_content)
            # Truncate the history after this message
            st.session_state.chat_history = st.session_state.chat_history[:idx + 1]

            # Hide the editor and trigger a resubmission
            st.session_state.show_editor = False
            handle_chat_submission() # Re-use the submission logic

    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.show_editor = False
            st.rerun()

# Standard chat input box, disabled if the editor is active
if user_query := st.chat_input("Please enter your question...", disabled=st.session_state.show_editor):
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    # Display the new user message immediately
    with st.chat_message("Human"):
        st.markdown(user_query)
    handle_chat_submission()
