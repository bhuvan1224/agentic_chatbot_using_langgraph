import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, TypedDict
from langchain_groq import ChatGroq
import smtplib
from email.message import EmailMessage

# Load environment variables from .env file
load_dotenv(override=True)

# ==============================================================================
# 1. PAGE SETUP & UI STYLING
# ==============================================================================
st.set_page_config(
    page_title="Agentic Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stChatFloatingInputContainer { padding-bottom: 20px; }
    
    .main-title {
        text-align: center;
        font-weight: 800;
        font-size: 2.3rem;
        margin-bottom: 0.3rem;
    }
    .sub-title {
        text-align: center;
        color: #9e9e9e;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    section[data-testid="stSidebar"] {
        background-color: #171717;
    }
    .stButton > button[kind="tertiary"] {
        justify-content: flex-start;
        padding-left: 10px;
        color: #d1d5db;
        border: none;
    }
    .stButton > button[kind="secondary"] {
        justify-content: flex-start;
        padding-left: 10px;
        background-color: #2f2f2f;
        color: #ffffff;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. LANGGRAPH AGENT SETUP (Robust Tools & Multi-LLM Resilience)
# ==============================================================================
@tool
def get_current_weather(location: str) -> str:
    """Get the current weather and rain conditions for a specific location."""
    return f"The weather in {location} is currently 28°C with clear skies and no immediate rain."

@tool
def get_future_weather(location: str, date: str) -> str:
    """Get the weather forecast for a future date."""
    return f"The forecast for {location} on {date} shows clear skies with a low chance of precipitation."

@tool
def get_stock_price(ticker: str) -> str:
    """Get the current market price of a stock, commodity, or asset (e.g. gold, AAPL, BTC)."""
    return f"The current market price of {ticker} is $2,650.00 per ounce/share."

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error calculating: {e}"

@tool
def web_search(query: str) -> str:
    """Search the web for general information or real-time news."""
    return f"Search results for '{query}': Real-time data retrieved successfully."

@tool
def send_email_report(recipient: str, subject: str, body_content: str) -> str:
    """Send a real email report using SMTP."""
    import os
    import smtplib
    from email.message import EmailMessage

    # Match these EXACT names with your .env file
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")

    if not sender_email or not sender_password:
        return "CRITICAL ERROR: SENDER_EMAIL or SENDER_PASSWORD missing in environment. Email NOT sent."

    try:
        msg = EmailMessage()
        msg.set_content(f"{body_content}\n\n---\nSent automatically by Agentic Chatbot")
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = recipient

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
            
        return f"SUCCESS: Email delivered to {recipient}."
    except Exception as e:
        return f"CRITICAL ERROR: Failed to send email: {str(e)}"

@tool
def purchase_stock(ticker: str, amount: float) -> str:
    """Buy shares of a stock. ONLY use this tool if the user explicitly commands you to buy stock."""
    return f"Successfully purchased {amount} shares of {ticker}."

@tool
def delete_vector_database() -> str:
    """Clear the RAG database. ONLY use this tool if the user explicitly commands you to delete the database."""
    return "Vector database successfully deleted."

# Move send_email_report to safe_tools
safe_tools = [get_current_weather, get_future_weather, get_stock_price, calculator, web_search, send_email_report]
sensitive_tools = [purchase_stock, delete_vector_database]
all_tools = safe_tools + sensitive_tools

# 1. Initialize models normally (DO NOT use .with_fallbacks() anymore)
primary_llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash", 
    temperature=0.2, 
    max_retries=1 # Fail fast so we can switch models immediately
).bind_tools(all_tools)

secondary_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    temperature=0.2, 
    max_retries=1
).bind_tools(all_tools)

try:
    from langchain_groq import ChatGroq
    groq_llm = ChatGroq(
        model="openai/gpt-oss-20b", 
        temperature=0.2, 
        max_retries=2
    ).bind_tools(all_tools)
except Exception:
    groq_llm = None

class State(TypedDict):
    messages: Annotated[list, add_messages]

sys_msg = SystemMessage(content="""You are an advanced, intelligent agentic AI assistant. 
- Think step-by-step and reason internally before answering.
- Use safe tools autonomously when needed to fetch information or complete routine tasks (like weather, web searches, and sending emails).
- NEVER use sensitive tools (like buying stocks or deleting databases) unless the user gives an explicit, direct command to do so.
- Respond in a clean, professional, human-readable format like ChatGPT or Claude.""")

# 2. The Ironclad Manual Failover Node
def chatbot_node(state: State):
    messages = [sys_msg] + state["messages"]
    
    try:
        # Attempt 1: Primary Gemini
        return {"messages": [primary_llm.invoke(messages)]}
    
    except Exception as primary_error:
        try:
            # Attempt 2: Secondary Gemini
            return {"messages": [secondary_llm.invoke(messages)]}
            
        except Exception as secondary_error:
            # Attempt 3: Groq (Ultimate Backup)
            if groq_llm is None:
                raise RuntimeError("Google models exhausted quotas, and Groq is not installed.")
                
            try:
                return {"messages": [groq_llm.invoke(messages)]}
            
            except Exception as groq_error:
                # If everything dies, expose exactly why Groq failed
                detailed_error = (
                    f"ALL MODELS FAILED.\n"
                    f"Gemini Issue: Rate limit / Quota exceeded.\n"
                    f"Groq Issue: {str(groq_error)}"
                )
                raise RuntimeError(detailed_error)

def route_tools(state: State):
    last_message = state["messages"][-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return END
    
    sensitive_names = [t.name for t in sensitive_tools]
    if any(tc["name"] in sensitive_names for tc in last_message.tool_calls):
        return "sensitive_tools"
    return "safe_tools"

@st.cache_resource
def get_checkpointer():
    return MemorySaver()

memory = get_checkpointer()

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_node("safe_tools", ToolNode(safe_tools))
graph_builder.add_node("sensitive_tools", ToolNode(sensitive_tools))

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", route_tools)
graph_builder.add_edge("safe_tools", "chatbot")
graph_builder.add_edge("sensitive_tools", "chatbot")

agent = graph_builder.compile(checkpointer=memory, interrupt_before=["sensitive_tools"])

# ==============================================================================
# 3. DYNAMIC SIDEBAR CHAT HISTORY
# ==============================================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

config = {"configurable": {"thread_id": st.session_state.thread_id}, "recursion_limit": 15}

with st.sidebar:
    if st.button("📝 New chat", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
        
    st.markdown("<br><p style='color: #888; font-size: 0.85em; font-weight: 600; margin-bottom: 8px;'>Recents</p>", unsafe_allow_html=True)
    
    if not st.session_state.chat_history:
        st.markdown("<p style='color: #555; font-size: 0.85em; padding-left: 10px;'>No previous chats yet.</p>", unsafe_allow_html=True)
    else:
        for tid, title in reversed(list(st.session_state.chat_history.items())):
            is_active = (tid == st.session_state.thread_id)
            btn_type = "secondary" if is_active else "tertiary"
            
            if st.button(f"💬 {title}", key=f"chat_{tid}", type=btn_type, use_container_width=True):
                st.session_state.thread_id = tid
                st.rerun()

# ==============================================================================
# 4. MAIN CHAT INTERFACE & CLEAN MESSAGE RENDERING
# ==============================================================================
st.markdown("<div class='main-title'>Agentic Chatbot</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Beyond basic QA — Intelligent reasoning, real-time tools, and human-governed actions.</div>", unsafe_allow_html=True)

current_state = agent.get_state(config)
messages = current_state.values.get("messages", []) if current_state.values else []

# STRICT FILTER: Render only human text and AI text. Completely hide tool logs.
for msg in messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"): 
            st.write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        if isinstance(msg.content, str) and msg.content.strip():
            with st.chat_message("assistant"): 
                st.write(msg.content)
        elif isinstance(msg.content, list):
            text_blocks = [b.get("text") for b in msg.content if isinstance(b, dict) and b.get("type") == "text"]
            if text_blocks and "".join(text_blocks).strip():
                with st.chat_message("assistant"):
                    st.write("".join(text_blocks))

is_paused = current_state.next and "sensitive_tools" in current_state.next

# ==============================================================================
# 5. HITL & BULLETPROOF EXECUTION
# ==============================================================================
if is_paused:
    last_msg = current_state.values["messages"][-1]
    st.markdown("---")
    st.warning("⚠️ **Action Required:** The assistant is attempting a sensitive action and requires your approval before proceeding.")
    
    for tc in last_msg.tool_calls:
        st.info(f"**Requested Tool:** `{tc['name']}` | **Parameters:** `{tc['args']}`")
        
    col1, col2 = st.columns([2, 2])
    with col1:
        if st.button("✅ Approve Action", type="primary", use_container_width=True):
            with st.spinner("Executing approved action..."):
                agent.invoke(None, config=config)
                st.rerun()
    with col2:
        if st.button("❌ Cancel Action", use_container_width=True):
            rejections = [
                ToolMessage(tool_call_id=tc["id"], name=tc["name"], content="Action cancelled by user.")
                for tc in last_msg.tool_calls
            ]
            agent.update_state(config, {"messages": rejections}, as_node="sensitive_tools")
            agent.invoke(None, config=config)
            st.rerun()
    st.markdown("---")
    st.info("🔒 Please click **Approve** or **Cancel** above to continue your conversation.")

else:
    chat_submission = st.chat_input(
        "Message your assistant...", 
        accept_file="multiple", 
        file_type=["pdf", "txt", "csv", "jpg", "png"]
    )

    if chat_submission:
        prompt_text = chat_submission.text
        uploaded_files = chat_submission.files if chat_submission.files else []

        if st.session_state.thread_id not in st.session_state.chat_history:
            clean_title = prompt_text.strip()
            title_snippet = clean_title[:28] + ("..." if len(clean_title) > 28 else "")
            st.session_state.chat_history[st.session_state.thread_id] = title_snippet or "New Conversation"

        with st.chat_message("user"): 
            st.write(prompt_text)
            for file in uploaded_files:
                st.caption(f"📎 Attached: {file.name}")

        if uploaded_files:
            file_names = ", ".join([f.name for f in uploaded_files])
            prompt_text = f"[User attached files: {file_names}]\n\n{prompt_text}"

        # Atomic execution using agent.invoke() inside a clean professional spinner
        with st.chat_message("assistant"):
            with st.spinner("Thinking and analyzing..."):
                try:
                    agent.invoke(
                        {"messages": [HumanMessage(content=prompt_text)]},
                        config=config
                    )
                except Exception as e:
                    # Print the error
                    st.error(f"⚠️ Service Notice: Unable to complete request. ({str(e)})")
                    # STOP the script here so it doesn't instantly rerun and erase the error!
                    st.stop() 
                    
        st.rerun()