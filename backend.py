import os
import math
import sqlite3
import smtplib
import requests
from typing import TypedDict, Annotated, Any
from email.message import EmailMessage

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# Import your custom nodes
from interviewer_node import AgentState, get_interviewer_node
from tailor_node import get_tailor_node

# ==========================================
# 1. LOAD ENVIRONMENT & INITIALIZE LLMS
# ==========================================
load_dotenv()

# Embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Initialize your individual models FIRST so they can be used below
primary_llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0.0
)

backup_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.0
)

# ==========================================
# 2. RESUME BUILDER GRAPH
# ==========================================

# Initialize nodes
interviewer = get_interviewer_node(primary_llm)
tailor = get_tailor_node(primary_llm)

def route_interviewer(state: AgentState):
    """Checks if the resume is complete. If yes, go to tailor. If no, pause for user input."""
    current_resume = state.get("resume_data")
    
    missing_fields = []
    if current_resume:
        for field, value in current_resume.model_dump().items():
            if not value: 
                missing_fields.append(field)
    
    if missing_fields:
        return "interviewer"
    else:
        return "tailor"

workflow = StateGraph(AgentState)
workflow.add_node("interviewer", interviewer)
workflow.add_node("tailor", tailor)
workflow.set_entry_point("interviewer")

# 💡 ROUTING MAP: 
# When route_interviewer returns "interviewer", send it to END (pause & wait for user).
# When route_interviewer returns "tailor", move to the tailor node.
workflow.add_conditional_edges(
    "interviewer",
    route_interviewer,
    {
        "interviewer": END,
        "tailor": "tailor"
    }
)

workflow.add_edge("tailor", END)

# 💡 DEFINE CHECKPOINT BEFORE COMPILING
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)

# Now Python knows what checkpoint is!
resume_bot = workflow.compile(checkpointer=checkpoint)


# ==========================================
# 3. TOOLS
# ==========================================

def ingest_rag_document(file_path):
    DB_PATH = "faiss_db"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(DB_PATH)
    

def get_retriever():
    DB_PATH = "faiss_db"
    vector_store = FAISS.load_local(
            folder_path=DB_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    return retriever


@tool
def rag_tool(query: str) -> str:
    """Retrieve relevant information from the PDF document."""
    retriever = get_retriever()
    documents = retriever.invoke(query)

    if not documents:
        return "No relevant information was found in the PDF."

    formatted_documents = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")
        formatted_documents.append(
            f"Document {index}\nSource: {source}\nPage: {page}\nContent: {document.page_content}"
        )

    return "\n\n".join(formatted_documents)

search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced"
)

@tool
def calculator(expression: str) -> str:
    """Useful for simple math calculations."""
    try:
        allowed = {"math": math, "abs": abs, "round": round, "min": min, "max": max, "sum": sum}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a given symbol."""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=9MZO2JUBR7IFNTOI"
    r = requests.get(url)
    return r.json()


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """Simulate purchasing a given quantity of a stock symbol (HITL)."""
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")

    if isinstance(decision, str) and decision.lower() == "yes":
        return {"status": "success", "message": f"Purchase order placed for {quantity} shares of {symbol}.", "symbol": symbol, "quantity": quantity}
    else:
        return {"status": "cancelled", "message": f"Purchase of {quantity} shares of {symbol} was declined by human.", "symbol": symbol, "quantity": quantity}


@tool
def get_current_weather(location: str) -> str:
    """Get the current real-time weather for a given city or location."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Weather API key is missing. Set the OPENWEATHER_API_KEY environment variable."

    try:
        geocoding_url = "https://api.openweathermap.org/geo/1.0/direct"
        geo_response = requests.get(geocoding_url, params={"q": location, "limit": 1, "appid": api_key}, timeout=10)
        geo_response.raise_for_status()
        locations: list[dict[str, Any]] = geo_response.json()

        if not locations:
            return f"Could not find the location: {location}"

        latitude = locations[0]["lat"]
        longitude = locations[0]["lon"]
        resolved_name = locations[0].get("name", location)
        
        weather_url = "https://api.openweathermap.org/data/2.5/weather"
        weather_response = requests.get(weather_url, params={"lat": latitude, "lon": longitude, "appid": api_key, "units": "metric"}, timeout=10)
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        return (f"Current weather in {resolved_name}:\n"
                f"- Condition: {weather_data['weather'][0]['description'].title()}\n"
                f"- Temperature: {weather_data['main']['temp']}°C\n")
    except Exception as error:
        return f"Weather service error: {error}"


@tool
def send_email_report(recipient: str, subject: str, body: str) -> dict:
    """Draft and send an email report to a specific recipient (HITL)."""
    decision = interrupt(f"Approve sending email to {recipient}? (yes/no)")
    if isinstance(decision, str) and decision.lower() == "yes":
        return {"status": "success", "message": f"Email successfully dispatched to {recipient}.", "final_body": body}
    return {"status": "rejected", "message": "Email broadcast was blocked or cancelled by human supervisor."}


@tool
def delete_vector_database() -> str:
    """Completely clear and delete the local RAG FAISS vector database (HITL)."""
    payload = {"action": "destructive_delete", "message": "WARNING: This will permanently delete your faiss_db vector index. Proceed?"}
    decision = interrupt(payload)

    if isinstance(decision, str) and decision.lower() == "yes":
        db_path = "faiss_db"
        if os.path.exists(db_path):
            import shutil
            shutil.rmtree(db_path)
            return "System database successfully wiped clean."
        return "Delete command executed, but no active database was detected."
    return "Database deletion aborted. Your indexed files are safe."


@tool
def send_email_tool(to_email: str, subject: str, body: str) -> str:
    """Sends an email to the specified recipient."""
    sender_email = os.environ.get("SENDER_EMAIL")
    sender_password = os.environ.get("SENDER_APP_PASSWORD")

    if not sender_email or not sender_password:
        return "Error: Sender email or password not configured in environment."

    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = to_email

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return f"Success! The email was successfully sent to {to_email}."
    except Exception as e:
        return f"Failed to send email. Error: {str(e)}"
    
@tool
def get_future_weather(location: str, days: int = 3) -> str:
    """Fetches future weather predictions and multi-day forecasts."""
    # NOTE: I swapped this to pull from your .env so you don't get the API error again!
    API_KEY = os.getenv("WEATHERAPI_KEY") 
    url = f"http://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={location}&days={days}&aqi=no&alerts=no"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if "error" in data:
            return f"⚠️ WeatherAPI Error for '{location}': {data['error'].get('message', 'Unknown error')}"
            
        summary = [f"🌤️ **Weather Forecast for {data['location']['name']}, {data['location']['country']}:**\n"]
        for day in data['forecast']['forecastday']:
            summary.append(f"📅 **{day['date']}**: {day['day']['condition']['text']} | Temp: {day['day']['mintemp_c']}°C to {day['day']['maxtemp_c']}°C")
        return "\n".join(summary)
    except Exception as e:
        return f"❌ System Error while fetching weather: {str(e)}"
        

# ==========================================
# 4. GENERAL CHATBOT GRAPH 
# ==========================================

tools = [
    search_tool, calculator, get_stock_price, get_current_weather, 
    rag_tool, purchase_stock, send_email_report, delete_vector_database,
    send_email_tool, get_future_weather
]

primary_with_tools = primary_llm.bind_tools(tools)
backup_with_tools = backup_llm.bind_tools(tools)
llm_with_tools = primary_with_tools.with_fallbacks([backup_with_tools])


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    """LLM node that can answer directly or call an appropriate tool."""
    system_message = SystemMessage(
    content=(
        "You are a helpful Agentic Chatbot.\n\n"
        "CRITICAL INSTRUCTION: For general knowledge, concepts, definitions, and history, YOU MUST ANSWER DIRECTLY FROM YOUR INTERNAL MEMORY. DO NOT CALL ANY TOOLS. ABSOLUTELY NO EXCEPTIONS.\n\n"
        "You are ONLY allowed to use tools under these strict conditions:\n"
        "1. `calculator`: USE ONLY for mathematical equations containing numbers.\n"
        "2. `rag_tool`: USE ONLY if the user asks a question about the contents of an uploaded document or PDF.\n"
        "3. `search_tool`: USE ONLY for live internet data like today's news, current events, or stock prices.\n"
        "4. `get_current_weather` / `get_future_weather`: USE ONLY for weather forecasts.\n\n"
        "*** SPECIAL RULE FOR 'RAG' ***\n"
        "If the user asks 'What is RAG?', they are asking for the definition of Retrieval-Augmented Generation. DO NOT use the `rag_tool`. Answer directly from your internal memory using a standard text response.\n"
    )
)
    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)

# Checkpointer
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)

graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)
graph.add_edge(START, 'chat_node')
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpoint)


# ==========================================
# 5. HELPER FUNCTIONS & CLI
# ==========================================

def get_all_threads():
    all_threads = set()
    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config['configurable']['thread_id'])
    return list(all_threads)

if __name__ == "__main__":
    print("🤖 Agentic Chatbot CLI\n")
    thread_id = "demo-thread"
    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in {"exit", "quit"}:
            break
        state = {"messages": [HumanMessage(content=user_input)]}
        result = chatbot.invoke(state, config={"configurable": {"thread_id": thread_id}})
        interrupts = result.get("__interrupt__", [])
        if interrupts:
            decision = input(f"HITL: {interrupts[0].value} (yes/no): ").strip().lower()
            result = chatbot.invoke(Command(resume=decision), config={"configurable": {"thread_id": thread_id}})
        print(f"Bot: {result['messages'][-1].content}\n")