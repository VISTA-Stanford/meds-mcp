"""
Secure LLM Client Interface

This module provides secure-llm Client interface for the MCP chat demo.
Uses secure-llm's native API directly.
"""

import logging
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

try:
    from securellm import Client, get_available_models as _securellm_get_available_models
    _SECURELLM_AVAILABLE = True
except ImportError:
    logger.error("securellm package not found. Install with: pip install secure-llm")
    Client = None
    _securellm_get_available_models = None
    _SECURELLM_AVAILABLE = False


# Global client instance (initialized once, reused)
_global_client = None


def get_client() -> Client:
    """
    Get or create the global secure-llm Client instance.
    
    Returns:
        secure-llm Client instance
    
    Raises:
        ImportError: If secure-llm is not installed
    """
    if not _SECURELLM_AVAILABLE:
        raise ImportError("securellm package not installed. Install with: pip install secure-llm")
    
    global _global_client
    if _global_client is None:
        _global_client = Client()
        logger.info("Initialized secure-llm Client")
    
    return _global_client


def get_llm_client(model_name: Optional[str] = None):
    """
    Get secure-llm Client instance.
    
    Args:
        model_name: Optional model identifier (for logging/compatibility).
                   Note: Model is specified per-request, not per-client.
    
    Returns:
        secure-llm Client instance
    """
    client = get_client()
    if model_name:
        logger.debug(f"Client ready for model: {model_name}")
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
    # secure-llm returns dict format: response["choices"][0]["message"]["content"]
    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if content:
                return content
    
    # Handle object-style response as fallback
    try:
        if hasattr(response, "choices") and response.choices:
            message = response.choices[0].message
            if hasattr(message, "content"):
                return message.content
            # Try dict access on object
            return response.choices[0]["message"]["content"]
    except (AttributeError, KeyError, IndexError):
        pass
    
    # Try direct dict access on object
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
        # Use secure-llm's built-in function to get models from registry
        models = _securellm_get_available_models()
        logger.debug(f"Retrieved {len(models)} models from secure-llm")
        return models
    except Exception as e:
        logger.error(f"Error getting available models from secure-llm: {e}")
        return []


def get_default_generation_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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

