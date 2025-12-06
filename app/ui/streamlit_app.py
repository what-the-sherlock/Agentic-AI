import time
from typing import Optional, Dict, Any
import requests
import streamlit as st

API_URL = "http://localhost:8000/api/agent"

st.set_page_config(page_title="Agent Smith", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Active Context Storage
if "active_text" not in st.session_state:
    st.session_state.active_text = None 

if "active_filename" not in st.session_state:
    st.session_state.active_filename = None

# Function to call the agent API
def call_agent_api(
    text: str,
    file_obj: Optional[st.runtime.uploaded_file_manager.UploadedFile],
    previous_context: Optional[str],
    context_text: Optional[str]
) -> Dict[str, Any]:
    data = {
        "text": text, 
        "previous_context": previous_context or "",
        "context_text": context_text or "" 
    }
    files = None

    if file_obj is not None:
        files = {
            "file": (
                file_obj.name,
                file_obj.getvalue(),
                file_obj.type or "application/octet-stream",
            )
        }

    resp = requests.post(API_URL, data=data, files=files, timeout=120)
    resp.raise_for_status()
    return resp.json()
# Sidebar for file upload
with st.sidebar:
    st.header("Attachments")
    st.caption("Upload a file to give the agent context.")
    
    uploaded_file = st.file_uploader(
        "File",
        type=["png", "jpg", "jpeg", "pdf", "mp3", "wav", "m4a"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        if uploaded_file.name != st.session_state.active_filename:
            st.session_state.active_text = None
            st.session_state.active_filename = uploaded_file.name
    
    if st.session_state.active_text:
        st.success(f"Context Active: {len(st.session_state.active_text)} chars")
        if st.button("Clear Context"):
            st.session_state.active_text = None
            st.session_state.active_filename = None
            st.rerun()

st.title("Agent Smith")

# For Displaying chat messages
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    st_role = "user" if role == "user" else "assistant"
    
    with st.chat_message(st_role):
        st.markdown(content)

        if role == "agent":
            meta = msg.get("meta", {})
            run_log = meta.get("run_log", {})
            if run_log and "cost_estimate" in run_log and run_log["cost_estimate"]:
                est = run_log["cost_estimate"]
                st.caption(
                    f"💰 **${est['total_cost_usd']:.6f}** | "
                    f"In: {est['input_tokens']} / Out: {est['output_tokens']}"
                )
# Chat input
if user_input := st.chat_input("Message..."):
    
    with st.chat_message("user"):
        st.markdown(user_input)

    previous_context_str = None
    if st.session_state.messages:
        previous_context_str = st.session_state.messages[-1]["content"]

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")

        try:
            file_to_send = uploaded_file if (uploaded_file and uploaded_file.name == st.session_state.active_filename) else None

            payload = call_agent_api(
                text=user_input,
                file_obj=file_to_send, 
                previous_context=previous_context_str,
                context_text=st.session_state.active_text 
            )

            status = payload.get("status")
            message = payload.get("message", "")
            extracted_text = payload.get("extracted_text")
            final_output = payload.get("final_output")
            run_log = payload.get("run_log", {})

            if extracted_text:
                st.session_state.active_text = extracted_text

            if status == "needs_clarification":
                    main_text = f"**Clarification Required:**\n\n{message}"
            elif status == "error":
                    main_text = f"**Error:**\n\n{message}"
            else:
                    main_text = final_output or "(No output)"

            streamed_text = ""
            for char in main_text:
                streamed_text += char
                if len(streamed_text) % 2 == 0: 
                    placeholder.markdown(f"{streamed_text}▌")
                    time.sleep(0.005) 
            
            placeholder.markdown(main_text)

            if run_log and "cost_estimate" in run_log and run_log["cost_estimate"]:
                est = run_log["cost_estimate"]
                st.caption(
                    f"💰 **${est['total_cost_usd']:.6f}** | "
                    f"In: {est['input_tokens']} / Out: {est['output_tokens']}"
                )

            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({
                "role": "agent", 
                "content": main_text, 
                "meta": {"run_log": run_log}
            })

        except Exception as e:
            placeholder.markdown(f"**Connection Error:** {e}")