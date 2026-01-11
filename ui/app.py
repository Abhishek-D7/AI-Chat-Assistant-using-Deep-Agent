import streamlit as st
import requests
import uuid

# Configuration
API_URL = "http://localhost:8000/chat"
USER_ID = "test_user_01" # Simulating a logged-in user

st.set_page_config(page_title="Deep Agent Chat", layout="wide")

st.title("🤖 Deep Agent AI Chat")

# Session State for Thread ID and Messages
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "debug_info" not in st.session_state:
    st.session_state.debug_info = {}

# Sidebar
with st.sidebar:
    st.header("Control Panel")
    if st.button("New Chat"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.debug_info = {}
        st.success("New Chat Started!")
    
    st.markdown(f"**Thread ID:** `{st.session_state.thread_id}`")
    
    st.header("Debug Info")
    if st.session_state.debug_info:
        with st.expander("Plan & Status", expanded=True):
            st.json(st.session_state.debug_info)
    else:
        st.info("Start chatting to see debug info.")

# Chat Area
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("How can I help you today?"):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API
    with st.spinner("Deep Agent is thinking (Plan -> Execute -> Review)..."):
        try:
            payload = {
                "message": prompt,
                "thread_id": st.session_state.thread_id,
                "user_id": USER_ID
            }
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            
            bot_reply = data["response"]
            plan = data["plan"]
            current_step = data["current_step"]
            
            # Update Debug Info
            st.session_state.debug_info = {
                "Plan": plan,
                "Current Step Index": current_step,
                "Task Complete": data["task_complete"]
            }
            
            # Add bot message to state
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            with st.chat_message("assistant"):
                st.markdown(bot_reply)
                
        except Exception as e:
            st.error(f"Error communicating with backend: {e}")
