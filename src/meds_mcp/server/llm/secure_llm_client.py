"""
Secure LLM Client interface for the MCP server.

Uses secure-llm's native API. Shared by cohort chat and other server endpoints.
"""

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

try:
    from securellm import Client, get_available_models as _securellm_get_available_models

    _SECURELLM_AVAILABLE = True
except ImportError:
    logger.error("securellm package not found. Install with: pip install secure-llm")
    Client = None
    _securellm_get_available_models = None
    _SECURELLM_AVAILABLE = False


_global_client = None


def get_client() -> Any:
    """
    Get or create the global secure-llm Client instance.

    Returns:
        secure-llm Client instance

    Raises:
        ImportError: If secure-llm is not installed
    """
    if not _SECURELLM_AVAILABLE:
        raise ImportError(
            "securellm package not installed. Install with: pip install secure-llm"
        )

    global _global_client
    if _global_client is None:
        _global_client = Client()
        logger.info("Initialized secure-llm Client")

    return _global_client


def get_llm_client(model_name: Optional[str] = None) -> Any:
    """
    Get secure-llm Client instance.

    Args:
        model_name: Optional model identifier (for logging/compatibility).
            Model is specified per-request, not per-client.

    Returns:
        secure-llm Client instance
    """
    client = get_client()
    if model_name:
        logger.debug("Client ready for model: %s", model_name)
    return client


def extract_response_content(response: Union[Dict[str, Any], Any]) -> str:
    """
    Extract content from secure-llm response.

    Args:
        response: Response from client.chat.completions.create()

    Returns:
        Response content text

    Raises:
        ValueError: If response format is unexpected
    """
    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if content:
                return content

    try:
        if hasattr(response, "choices") and response.choices:
            message = response.choices[0].message
            if hasattr(message, "content"):
                return message.content
            return response.choices[0]["message"]["content"]
    except (AttributeError, KeyError, IndexError):
        pass

    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        pass

    raise ValueError(f"Unexpected response format: {type(response)}")


def get_available_models() -> List[str]:
    """
    Get list of available models from secure-llm.

    Returns:
        List of model identifiers from secure-llm's model registry
    """
    if not _SECURELLM_AVAILABLE or _securellm_get_available_models is None:
        logger.warning("securellm not available, returning empty model list")
        return []

    try:
        models = _securellm_get_available_models()
        logger.debug("Retrieved %d models from secure-llm", len(models))
        return models
    except Exception as e:
        logger.error("Error getting available models from secure-llm: %s", e)
        return []


def get_default_generation_config(
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Get default generation configuration.

    Args:
        overrides: Dict of parameters to override defaults

    Returns:
        Generation config dict
    """
    defaults = {
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 2048,
    }

    if overrides:
        defaults.update(overrides)

    return defaults
