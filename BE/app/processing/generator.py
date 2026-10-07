import json
import requests
from typing import List, Dict, Any
from app.core.config import GEMINI_API_KEY, GROQ_API_KEY
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Initialize Gemini LLM if key is valid
llm_gemini = None
if GEMINI_API_KEY and not GEMINI_API_KEY.startswith("your_"):
    try:
        llm_gemini = ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.3
        )
    except Exception:
        llm_gemini = None

RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Use the following pieces of retrieved context to answer the question.
If you don't know the answer based on the context, just say that you don't know. Do not try to make up an answer.
Keep the answer clear and concise.

Context:
{context}

Question: {question}

Answer:"""

prompt_template = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


def generate_with_groq(prompt_text: str) -> str:
    """
    Generates response using Groq Cloud API (openai/gpt-oss-120b).
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0.3
    }

    res = requests.post(url, headers=headers, json=payload, timeout=30)
    if res.status_code == 200:
        data = res.json()
        return data["choices"][0]["message"]["content"]
    else:
        raise RuntimeError(f"Groq API Error ({res.status_code}): {res.text}")


def generate_rag_answer(query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Generates RAG answer and returns both the answer and the LLM provider name.
    """
    if not retrieved_chunks:
        return {
            "answer": "I couldn't find any relevant information in the uploaded documents to answer your question.",
            "provider": "None"
        }

    # Combine the chunks into a single context string
    context_text = "\n\n---\n\n".join([chunk["content"] for chunk in retrieved_chunks])
    formatted_prompt = prompt_template.format(context=context_text, question=query)

    # 1. Try Groq Cloud API first (ultra-fast LLM inference)
    if GROQ_API_KEY:
        try:
            answer = generate_with_groq(formatted_prompt)
            return {
                "answer": answer,
                "provider": "Groq Cloud (openai/gpt-oss-120b)"
            }
        except Exception as groq_err:
            print(f"[Warning] Groq LLM failed: {groq_err}. Trying fallback to Gemini...")

    # 2. Fallback to Gemini LLM if available
    if llm_gemini:
        try:
            response = llm_gemini.invoke(formatted_prompt)
            return {
                "answer": str(response.content),
                "provider": "Google Gemini (gemini-1.5-flash)"
            }
        except Exception as gemini_err:
            return {
                "answer": f"An error occurred while generating answer: {str(gemini_err)}",
                "provider": "Error"
            }

    return {
        "answer": "Error: Neither GROQ_API_KEY nor valid GEMINI_API_KEY is configured.",
        "provider": "Error"
    }
