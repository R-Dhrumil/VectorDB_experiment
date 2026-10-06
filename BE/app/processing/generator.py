from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from app.core.config import GEMINI_API_KEY
from typing import List, Dict, Any

# Ensure we have the API key
if GEMINI_API_KEY:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3
    )
else:
    llm = None

RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Use the following pieces of retrieved context to answer the question.
If you don't know the answer based on the context, just say that you don't know. Do not try to make up an answer.
Keep the answer clear and concise.

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

def generate_rag_answer(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    if not llm:
        return "Error: GEMINI_API_KEY is not configured in the environment."
        
    if not retrieved_chunks:
        return "I couldn't find any relevant information in the uploaded documents to answer your question."
        
    # Combine the chunks into a single context string
    context_text = "\n\n---\n\n".join([chunk["content"] for chunk in retrieved_chunks])
    
    # Format the prompt
    formatted_prompt = prompt.format(context=context_text, question=query)
    
    try:
        # Generate the answer using Gemini
        response = llm.invoke(formatted_prompt)
        return str(response.content)
    except Exception as e:
        return f"An error occurred while generating the answer: {str(e)}"
