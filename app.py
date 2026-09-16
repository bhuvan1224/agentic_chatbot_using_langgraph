import streamlit as st
# ========================= Page configuration =========================
# THIS MUST BE THE ABSOLUTE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Agentic Chatbot",
    page_icon="🤖"
)

import uuid
import tempfile
import os
import json
import io
import markdown
from xhtml2pdf import pisa

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage
)
from langgraph.types import Command

from backend import (
    chatbot,
    resume_bot,
    get_all_threads,
    ingest_rag_document,
    primary_llm 
)

def generate_pdf_from_md(md_text):
    """Converts Markdown text into a strict 1-Page ATS-friendly PDF."""
    raw_html = markdown.markdown(md_text)
    
    styled_html = f"""
    <html>
    <head>
    <style>
        @page {{ 
            size: letter portrait;
            margin: 1.2cm 1.2cm 1.2cm 1.2cm; /* Strict 0.5 inch margins */
        }}
        body {{ 
            font-family: 'Helvetica', 'Arial', sans-serif; 
            font-size: 9pt; /* Smaller font to guarantee fit */
            color: #000000; 
            line-height: 1.1; 
        }}
        h1 {{ 
            font-size: 16pt; 
            text-align: center; 
            margin: 0 0 2px 0; 
            text-transform: uppercase;
        }}
        h2 {{ 
            font-size: 10.5pt; 
            border-bottom: 1px solid #000000; 
            padding-bottom: 2px; 
            margin-top: 6px; 
            margin-bottom: 4px; 
            text-transform: uppercase;
        }}
        p {{ margin: 1px 0; }}
        ul {{ margin-top: 1px; margin-bottom: 3px; padding-left: 12px; }}
        li {{ margin-bottom: 1px; text-align: justify; }}
    </style>
    </head>
    <body>
        {raw_html}
    </body>
    </html>
    """
    
    pdf_buffer = io.BytesIO()
    pisa.CreatePDF(io.StringIO(styled_html), dest=pdf_buffer)
    return pdf_buffer.getvalue()

TITLES_FILE = "chat_titles.json"

# ========================= Helper Functions =========================

def format_message_content(content) -> str:
    """Cleans up raw LLM output, extracting only the readable text."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    text_parts.append(item["text"])
                elif "text" in item:
                    text_parts.append(str(item["text"]))
        return "".join(text_parts).strip()
    return str(content) if content else ""


def get_chat_title(thread_id):
    if os.path.exists(TITLES_FILE):
        try:
            with open(TITLES_FILE, "r") as f:
                titles = json.load(f)
                if thread_id in titles and titles[thread_id]:
                    return titles[thread_id]
        except Exception:
            pass

    config = {"configurable": {"thread_id": thread_id}}
    state = chatbot.get_state(config)

    if not state or not state.values or "messages" not in state.values:
        return "New Chat"

    first_user_msg = None
    for msg in state.values["messages"]:
        if isinstance(msg, HumanMessage):
            first_user_msg = format_message_content(msg.content)
            break

    if not first_user_msg or not first_user_msg.strip():
        return "New Chat"

    try:
        prompt = (
            "Generate a extremely short, 1 to 3 word title summarizing the user's topic. "
            "Do NOT use quotes, punctuation, preamble, or markdown. Examples:\n"
            "Prompt: What is the weather in Delhi? -> Delhi Weather\n"
            "Prompt: Tell me the latest news -> World News\n"
            "Prompt: Hey how are you? -> Greeting\n\n"
            f"User Prompt: {first_user_msg}\nTitle:"
        )
        
        response = primary_llm.invoke(prompt)
        raw_title = format_message_content(response.content)
        
        title = raw_title.replace('"', '').replace("'", "").strip().split('\n')[0]
        if not title:
            title = first_user_msg[:20].strip()
    except Exception:
        title = first_user_msg[:20].strip()

    titles = {}
    if os.path.exists(TITLES_FILE):
        try:
            with open(TITLES_FILE, "r") as f:
                titles = json.load(f)
        except Exception:
            pass

    titles[thread_id] = title
    with open(TITLES_FILE, "w") as f:
        json.dump(titles, f)

    return title


def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if "chat_threads" not in st.session_state:
        st.session_state["chat_threads"] = []
        
    if thread_id in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].remove(thread_id)
        
    st.session_state["chat_threads"].insert(0, thread_id)


def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["pending_hitl"] = None
    add_thread(st.session_state["thread_id"])


def delete_thread(thread_id):
    """Removes a thread from the UI and clears its saved title."""
    if thread_id in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].remove(thread_id)
        
    if os.path.exists(TITLES_FILE):
        try:
            with open(TITLES_FILE, "r") as f:
                titles = json.load(f)
            if thread_id in titles:
                del titles[thread_id]
                with open(TITLES_FILE, "w") as f:
                    json.dump(titles, f)
        except Exception:
            pass
            
    if st.session_state.get("thread_id") == thread_id:
        reset_chat()


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )
    return state.values.get("messages", [])


def get_pending_interrupt(thread_id):
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:
        state_snapshot = chatbot.get_state(config)

        direct_interrupts = getattr(state_snapshot, "interrupts", ()) or ()
        if direct_interrupts:
            return direct_interrupts[0]

        tasks = getattr(state_snapshot, "tasks", ()) or ()
        for task in tasks:
            task_interrupts = getattr(task, "interrupts", ()) or ()
            if task_interrupts:
                return task_interrupts[0]

    except Exception:
        return None

    return None


def save_pending_interrupt(thread_id, interrupt_object):
    st.session_state["pending_hitl"] = {
        "thread_id": thread_id,
        "prompt": str(interrupt_object.value)
    }


def sync_pending_interrupt(thread_id):
    pending_interrupt = get_pending_interrupt(thread_id)

    if pending_interrupt is not None:
        save_pending_interrupt(thread_id, pending_interrupt)
    else:
        current_pending = st.session_state.get("pending_hitl")
        if current_pending is not None and current_pending.get("thread_id") == thread_id:
            st.session_state["pending_hitl"] = None


def resume_hitl_execution(decision):
    pending_hitl = st.session_state.get("pending_hitl")

    if not pending_hitl:
        st.warning("There is no pending action to approve or reject.")
        return

    interrupted_thread_id = pending_hitl["thread_id"]

    resume_config = {
        "configurable": {"thread_id": interrupted_thread_id},
        "metadata": {"thread_id": interrupted_thread_id},
        "run_name": "hitl_resume_trace",
    }

    try:
        with st.chat_message("assistant"):
            status_holder = {
                "box": st.status("🔄 Resuming the requested action...", expanded=True)
            }

            def resumed_ai_only_stream():
                for message_chunk, metadata in chatbot.stream(
                    Command(resume=decision),
                    config=resume_config,
                    stream_mode="messages",
                ):
                    if isinstance(message_chunk, ToolMessage):
                        tool_name = getattr(message_chunk, "name", "tool")
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                    if isinstance(message_chunk, AIMessage):
                        clean_text = format_message_content(message_chunk.content)
                        if clean_text:
                            yield clean_text

            st.write_stream(resumed_ai_only_stream())
            next_interrupt = get_pending_interrupt(interrupted_thread_id)

            if next_interrupt is not None:
                save_pending_interrupt(interrupted_thread_id, next_interrupt)
                status_holder["box"].update(
                    label="⚠️ Another approval is required",
                    state="complete",
                    expanded=False
                )
            else:
                st.session_state["pending_hitl"] = None
                status_holder["box"].update(
                    label="✅ Action completed",
                    state="complete",
                    expanded=False
                )

        st.rerun()

    except Exception as error:
        st.error(f"Could not resume the requested action: {error}")


# ========================= Main App Initialization =========================

st.title("Agentic Chatbot with LangGraph")

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

if "pending_hitl" not in st.session_state:
    st.session_state["pending_hitl"] = None

add_thread(st.session_state["thread_id"])
sync_pending_interrupt(st.session_state["thread_id"])


# ========================= Sidebar threading feature =========================

st.sidebar.title("My Conversations")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.markdown("---")
st.session_state["bot_mode"] = st.sidebar.selectbox(
    "🤖 Choose AI Agent:", 
    ["General Assistant", "Resume Builder"]
)

st.sidebar.markdown("---")

for thread_id in list(st.session_state["chat_threads"]):
    display_name = get_chat_title(thread_id)

    if thread_id == st.session_state["thread_id"]:
        button_label = f"💬 {display_name}"
    else:
        button_label = f"📄 {display_name}"

    col1, col2 = st.sidebar.columns([4, 1])

    with col1:
        if st.button(button_label, key=f"select_{thread_id}", use_container_width=True):
            st.session_state["thread_id"] = thread_id
            add_thread(thread_id)
            sync_pending_interrupt(thread_id)
            st.rerun()

    with col2:
        if st.button("🗑️", key=f"delete_{thread_id}", use_container_width=True):
            delete_thread(thread_id)
            st.rerun()

# ========================= HITL approval interface =========================

pending_hitl = st.session_state.get("pending_hitl")
current_thread_has_pending_hitl = (
    pending_hitl is not None
    and pending_hitl.get("thread_id") == st.session_state["thread_id"]
)

if current_thread_has_pending_hitl:
    st.warning(
        "🧑 **Human approval required**\n\n"
        f"{pending_hitl['prompt']}"
    )

    approve_column, reject_column = st.columns(2)

    with approve_column:
        if st.button(
            "✅ Approve Action",
            key=f"approve_{st.session_state['thread_id']}",
            type="primary",
            use_container_width=True
        ):
            resume_hitl_execution("yes")

    with reject_column:
        if st.button(
            "❌ Reject Action",
            key=f"reject_{st.session_state['thread_id']}",
            use_container_width=True
        ):
            resume_hitl_execution("no")


# ========================= Fixed chat input with PDF upload =========================

CONFIG = {
    "configurable": {"thread_id": st.session_state["thread_id"]},
    "metadata": {"thread_id": st.session_state["thread_id"]},
    "run_name": "chat_trace",
}

current_mode = st.session_state.get("bot_mode", "General Assistant")
active_bot = chatbot if current_mode == "General Assistant" else resume_bot

# Fetch the exact truth from the database and clean dictionary data chunks
try:
    history_state = active_bot.get_state(CONFIG)
    if "messages" in history_state.values:
        for msg in history_state.values["messages"]:
            
            if type(msg).__name__ == "HumanMessage":
                clean_text = format_message_content(msg.content)
                if clean_text:
                    with st.chat_message("user"):
                        st.markdown(clean_text)
                        
            elif type(msg).__name__ == "AIMessage":
                clean_text = format_message_content(msg.content)
                # Hide raw tool calls like <RAG=rag_tool> from old messages
                if clean_text and not ("<RAG=" in clean_text or '{"query"' in clean_text or "<search=" in clean_text):
                    with st.chat_message("assistant"):
                        st.markdown(clean_text)
except Exception:
    pass # Ignore if the thread is brand new and empty


submission = st.chat_input(
    "Type here",
    accept_file=True,
    file_type=["pdf"],
    disabled=bool(current_thread_has_pending_hitl)
)

if submission:
    add_thread(st.session_state["thread_id"])
    
    user_input = submission.text
    uploaded_files = submission.files

    if uploaded_files:
        uploaded_pdf = uploaded_files[0]
        temporary_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temporary_file:
                temporary_file.write(uploaded_pdf.getvalue())
                temporary_file_path = temporary_file.name

            with st.spinner(f"Processing {uploaded_pdf.name}..."):
                ingest_rag_document(temporary_file_path)

            st.toast(f"{uploaded_pdf.name} processed successfully.", icon="✅")
        except Exception as error:
            st.error(f"PDF processing failed: {error}")
        finally:
            if temporary_file_path and os.path.exists(temporary_file_path):
                os.remove(temporary_file_path)

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            if current_mode == "General Assistant":
                status_holder = {"box": None}

                def ai_only_stream():
                    for message_chunk, metadata in chatbot.stream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        if isinstance(message_chunk, ToolMessage):
                            tool_name = getattr(message_chunk, "name", "tool")
                            if status_holder["box"] is None:
                                status_holder["box"] = st.status(f"🔧 Using `{tool_name}` …", expanded=True)
                            else:
                                status_holder["box"].update(label=f"🔧 Using `{tool_name}` …", state="running", expanded=True)

                        if isinstance(message_chunk, AIMessage):
                            # Ensure the streaming chunks pass through the cleaner too!
                            clean_text = format_message_content(getattr(message_chunk, "content", ""))
                            
                            if clean_text and not ("<RAG=" in clean_text or '{"query"' in clean_text or "<search=" in clean_text):
                                yield clean_text

                    pending_interrupt = get_pending_interrupt(st.session_state["thread_id"])
                    if pending_interrupt is not None:
                        save_pending_interrupt(st.session_state["thread_id"], pending_interrupt)
                        yield "\n\n⚠️ **This action requires confirmation.**"

                st.write_stream(ai_only_stream())

                if status_holder["box"] is not None:
                    if get_pending_interrupt(st.session_state["thread_id"]) is not None:
                        status_holder["box"].update(label="⏸️ Waiting for human approval", state="complete", expanded=False)
                    else:
                        status_holder["box"].update(label="✅ Tool finished", state="complete", expanded=False)

            else:
                with st.spinner("Analyzing profile..."):
                    result = resume_bot.invoke({"messages": [HumanMessage(content=user_input)]}, config=CONFIG)
                    messages = result.get("messages", [])
                    
                    raw_ai_message = messages[-1].content if messages else "Error processing resume data."
                    clean_ai_message = format_message_content(raw_ai_message)
                    st.markdown(clean_ai_message)

                    current_state = resume_bot.get_state(CONFIG).values
                    if "tailored_resume" in current_state and current_state["tailored_resume"]:
                        pdf_bytes = generate_pdf_from_md(current_state["tailored_resume"])
                        st.download_button(
                            label="📄 Download Tailored Resume (PDF)",
                            data=pdf_bytes,
                            file_name="Bhuvaneswar_Resume.pdf",
                            mime="application/pdf"
                        )
        
        st.rerun()