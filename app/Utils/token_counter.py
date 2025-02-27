import tiktoken

def count_tokens(text: str, model_name: str) -> int:
    """Counts tokens dynamically based on the selected model."""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))
    except Exception:
        return len(text.split())  # Approximate token count if model is unknown
