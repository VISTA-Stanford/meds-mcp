import tiktoken


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base if model not found
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))
