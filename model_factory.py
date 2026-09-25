import os
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel

def get_chat_model(
    provider: str,
    model_name: str,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    base_url: Optional[str] = None
) -> BaseChatModel:
    """Factory to instantiate any LLM provider dynamically.
    
    Includes max_retries=0 across providers to fail immediately on rate limits 
    or quota errors instead of causing long background retries.
    """
    provider = provider.lower() if provider else ""

    if "openai" in provider:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name or "gpt-4o-mini",
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            temperature=temperature,
            max_retries=0
        )

    elif "anthropic" in provider or "claude" in provider:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model_name=model_name or "claude-3-5-sonnet-latest",
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            temperature=temperature,
            max_retries=0
        )

    elif "groq" in provider:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model_name or "llama-3.3-70b-versatile",
            api_key=api_key or os.getenv("GROQ_API_KEY"),
            temperature=temperature,
            max_retries=0
        )

    elif "ollama" in provider:
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            model=model_name or "llama3",
            base_url=base_url or "http://localhost:11434",
            temperature=temperature
        )

    else:  # Default: Google Gemini
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Updated to the currently supported Flash model with the large free tier
        selected_model = model_name if model_name else "gemini-3.8-flash"
        return ChatGoogleGenerativeAI(
            model=selected_model,
            google_api_key=api_key or os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
            max_retries=0
        )