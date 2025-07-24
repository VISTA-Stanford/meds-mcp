"""
Chat and LLM interaction functions for the MCP Chat Demo.
"""

import asyncio
import logging
import re
from typing import List, Dict, Any

import pandas as pd

from chat.utils.tokens import count_tokens
from chat.core.session import session_state
from chat.visualization.timeline import TimelineManager
from chat.mcp_client.client import search_patient_events_simple
from lumia.engines import get_llm_client, get_default_generation_config

logger = logging.getLogger(__name__)


def pack_context_for_query(
    system_prompt: str,
    history: List[Dict[str, str]],
    user_input: str,
    max_context_len: int,
    timeline_mode: bool,
    prompt_template: str,
    mcp_url: str,
):
    """Pack context for query using MCP server search or timeline mode."""

    if not session_state.timeline_loaded or not session_state.current_patient_id:
        logger.warning("❌📦 No patient data loaded")
        return [], None

    logger.info(f"📦 Packing context for patient {session_state.current_patient_id}")
    logger.info(f"📦 Timeline mode: {timeline_mode}, Max context: {max_context_len}")

    # Get events based on mode
    if timeline_mode:
        # Use all events in reverse chronological order, filtered by query datetime
        relevant_events = session_state.patient_events.copy()

        # Filter by query datetime if set
        if session_state.query_datetime is not None:
            filtered_events = []
            query_dt = pd.to_datetime(session_state.query_datetime)

            for event in relevant_events:
                metadata = event.get("metadata", {})
                if "timestamp" in metadata:
                    try:
                        event_time = pd.to_datetime(metadata["timestamp"])
                        if event_time <= query_dt:
                            filtered_events.append(event)
                    except (ValueError, TypeError):
                        continue

            relevant_events = filtered_events
            logger.info(
                f"📦 Filtered to {len(relevant_events)} events by query datetime"
            )

        # Sort by timestamp (reverse chronological)
        relevant_events.sort(
            key=lambda x: pd.to_datetime(
                x.get("metadata", {}).get("timestamp", "1970-01-01")
            ),
            reverse=True,
        )

    else:
        # Use search
        try:
            relevant_events = asyncio.run(
                search_patient_events_simple(
                    user_input, session_state.current_patient_id, mcp_url
                )
            )
            logger.info(f"📦 Search returned {len(relevant_events)} events")
        except Exception as e:
            logger.error(f"Error in search: {e}")
            relevant_events = session_state.patient_events[:20]  # Fallback

    # Construct messages list
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt.strip()})

    # Add chat history
    if history:
        for msg in history:
            if isinstance(msg, dict):
                messages.append(msg)
            else:  # Handle old tuple format
                user_msg, assistant_msg = msg
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": assistant_msg})

    # Process events to create context
    processed_events = []
    for event in relevant_events:
        content = event.get("content", "")
        metadata = event.get("metadata", {})

        # Add timestamp if available
        if "timestamp" in metadata and "timestamp=" not in content:
            try:
                timestamp = pd.to_datetime(metadata["timestamp"]).strftime("%B %d, %Y")
                content = f"[{timestamp}] {content}"
            except (ValueError, TypeError):
                pass

        processed_events.append(content)

    # Combine context and create final message
    full_context = "\n\n".join(processed_events)
    combined_message = prompt_template.format(context=full_context, question=user_input)
    messages.append({"role": "user", "content": combined_message})

    # Check token length and truncate if needed
    prompt_tokens = sum(count_tokens(m["content"]) for m in messages)
    logger.info(f"📦 Total tokens in prompt: {prompt_tokens}")

    final_events = relevant_events.copy()
    if prompt_tokens > max_context_len:
        # Truncate context
        pruned_events = []
        token_count = sum(count_tokens(m["content"]) for m in messages[:-1])

        for i, event in enumerate(relevant_events):
            event_tokens = count_tokens(processed_events[i])
            if token_count + event_tokens < max_context_len:
                pruned_events.append(processed_events[i])
                token_count += event_tokens
            else:
                break

        final_events = relevant_events[: len(pruned_events)]
        full_context = "\n\n".join(pruned_events)
        logger.info(f"📦 Pruned to {len(pruned_events)} events due to token limit")

        # Update the final message
        messages[-1] = {
            "role": "user",
            "content": prompt_template.format(
                context=full_context, question=user_input
            ),
        }

    # Update timeline plot with highlighted events
    fig = TimelineManager.update_timeline_plot(final_events)

    return messages, fig


def stream_chat_response(
    user_input: str,
    history: List[Dict[str, str]],
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeline_mode: bool,
    model_name: str,
    max_input_length: int,
    prompt_template: str,
    use_cache: bool,
    mcp_url: str,
    llm_client,
):
    """Generate responses from the LLM using vanilla generate method."""
    if history is None:
        history = []

    # Update LLM client if model changed
    if model_name != llm_client.model_name:
        logger.info(f"Switching LLM model from {llm_client.model_name} to {model_name}")
        llm_client = get_llm_client(model_name)

    # Pack context
    try:
        messages, fig = pack_context_for_query(
            system_prompt,
            history,
            user_input,
            max_input_length,
            timeline_mode,
            prompt_template,
            mcp_url,
        )
    except Exception as e:
        logger.error(f"Error packing context: {e}")
        messages = []
        fig = None

    if not messages:
        reply = "[No patient data loaded. Please load a patient first.]"
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})
        return history, history, fig

    # Add user message to history
    history.append({"role": "user", "content": user_input})

    # Create generation config
    generation_config = get_default_generation_config(
        {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            # Note: max_input_length is used locally for context truncation, not sent to LLM API
        }
    )

    try:
        # Generate response using vanilla generate method
        logger.info("🤖 Generating response...")
        response = llm_client.generate(
            messages=messages,
            generation_config=generation_config,
        )

        logger.info("✅ Response generated successfully")

        # Parse JSON response if present
        try:
            import json

            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                response_data = json.loads(json_str)
                logger.info("=== Raw JSON Response ===")
                logger.info(json.dumps(response_data, indent=2))
                logger.info("=======================")

                formatted_response = f"{response_data['answer']}\n\nEvidence UIDs: {', '.join(response_data['evidence'])}"
                history.append({"role": "assistant", "content": formatted_response})
            else:
                # No JSON found, use raw response
                history.append({"role": "assistant", "content": response})

        except Exception as e:
            logger.error(f"Error parsing JSON response: {str(e)}")
            # Fallback to raw response
            history.append({"role": "assistant", "content": response})

        return history, history, fig

    except Exception as e:
        logger.error(f"Error during generation: {str(e)}", exc_info=True)
        history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        return history, history, fig
