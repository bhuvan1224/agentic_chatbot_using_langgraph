# Agentic AI Chatbot & ATS Resume Builder

## About the Project
An advanced, stateful Agentic AI system designed to handle complex, multi-step workflows using LangGraph and Python. Built with a dual-agent architecture, the application serves as both a highly capable General Assistant and a specialized ATS Resume Builder.

The system utilizes Large Language Models (Gemini and Groq) and dynamically routes user queries to specialized tools, including FAISS-based RAG for PDF document analysis, web search for real-time data retrieval, and mathematical calculation tools. A core feature of the platform is its autonomous Resume Builder, which parses conversational inputs to extract structured data and generates 1-page, ATS-compliant PDF resumes tailored to specific job descriptions.

To ensure enterprise-level reliability, the architecture integrates a Human-in-the-Loop (HITL) approval system for sensitive actions and utilizes SQLite checkpointers for persistent, cross-session memory management.

## Interface & Key Features
*   **Dual-Agent Architecture:** Seamlessly switch between a "General Assistant" and a "Resume Builder" via the Streamlit sidebar[cite: 1].
*   **Intelligent Tool Routing:** The general agent autonomously uses tools like Tavily Search, Weather APIs, a Calculator, and FAISS RAG based on the context of the user's prompt[cite: 2].
*   **Automated Resume Generation:** The Resume Builder uses a state machine loop to interview the user, extract structured data using strict validation, and compile a compact, 1-page ATS-friendly PDF[cite: 1, 6, 8].
*   **Persistent Memory:** Powered by LangGraph's SQLite checkpointer (`chatbot.db`), allowing users to save, resume, switch between, and delete previous chat threads natively within the UI[cite: 1, 2].
*   **Human-in-the-Loop (HITL):** Sensitive tools (like deleting the vector database or sending automated emails) pause the execution graph and require explicit human approval via the UI before proceeding[cite: 1, 2].
*   **Clean Chat UI:** Implements custom message extraction to prevent raw JSON/dictionary metadata from leaking into the chat interface[cite: 1].

## Tech Stack
*   **Frontend UI:** Streamlit[cite: 1]
*   **Agentic Framework:** LangGraph, LangChain[cite: 2]
*   **Large Language Models:** Groq (Llama 3.1 8B), Google Generative AI (Gemini 2.5 Flash, Gemini Embeddings)[cite: 2]
*   **RAG & Vector Storage:** FAISS, PyPDFLoader, RecursiveCharacterTextSplitter[cite: 2]
*   **Memory & Database:** SQLite[cite: 2]
*   **Document Generation:** `markdown`, `xhtml2pdf`[cite: 1]
*   **Data Validation:** Pydantic[cite: 6]

## How to Run Locally
1. Clone the repository to your local machine.
2. Install the required dependencies: `pip install -r requirements.txt`
3. Create a `.env` file in the root directory and add your API keys:
   * `GROQ_API_KEY`
   * `GOOGLE_API_KEY`
   * `TAVILY_API_KEY`
   * `OPENWEATHER_API_KEY`
   * `WEATHERAPI_KEY`
4. Run the application: `streamlit run app.py`
