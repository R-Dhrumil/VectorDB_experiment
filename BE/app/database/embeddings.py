import httpx
from typing import List, Union, overload
from app.core.config import OLLAMA_BASE_URL, DEFAULT_EMBEDDING_MODEL


@overload
def get_embedding(
    text: str, 
    model: str = DEFAULT_EMBEDDING_MODEL, 
    is_query: bool = False
) -> List[float]: ...


@overload
def get_embedding(
    text: List[str], 
    model: str = DEFAULT_EMBEDDING_MODEL, 
    is_query: bool = False
) -> List[List[float]]: ...


def get_embedding(
    text: Union[str, List[str]], 
    model: str = DEFAULT_EMBEDDING_MODEL,
    is_query: bool = False
) -> Union[List[float], List[List[float]]]:
    """
    Generate vector embeddings using Ollama (nomic-embed-text by default).
    
    nomic-embed-text uses task prefixes:
      - 'search_document: ' for chunks/documents being stored
      - 'search_query: ' for user search queries
      
    :param text: A single text string or list of text strings.
    :param model: The Ollama embedding model name.
    :param is_query: True for search queries, False for documents.
    :return: 768-dimensional float list (or list of lists).
    """
    is_single = isinstance(text, str)
    texts = [text] if is_single else text

    # Apply task prefix if using nomic-embed-text
    if "nomic" in model.lower():
        prefix = "search_query: " if is_query else "search_document: "
        formatted_input = [f"{prefix}{t}" for t in texts]
    else:
        formatted_input = texts

    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/embed"
    payload = {
        "model": model,
        "input": formatted_input
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings", [])
            if not embeddings:
                raise ValueError("Ollama returned an empty embeddings array.")
            return embeddings[0] if is_single else embeddings
    except httpx.ConnectError:
        raise ConnectionError(
            f"Could not connect to Ollama at '{OLLAMA_BASE_URL}'. "
            "Please make sure Ollama is running ('ollama serve' or open Ollama app) "
            f"and you have downloaded the model ('ollama pull {model}')."
        )
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Ollama returned HTTP {e.response.status_code}: {e.response.text}")
    except Exception as e:
        raise RuntimeError(f"Failed to generate embedding with {model}: {str(e)}")
