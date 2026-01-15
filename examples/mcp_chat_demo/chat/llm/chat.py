"""
Chat and LLM interaction functions for the MCP Chat Demo.
"""

import asyncio
import json
import logging
import re
from typing import List, Dict, Any
from datetime import datetime

import pandas as pd

from chat.utils.tokens import count_tokens
from chat.core.session import session_state
from chat.visualization.timeline import TimelineManager
from chat.mcp_client.client import search_patient_events_simple
from chat.llm.secure_llm_client import get_llm_client, extract_response_content

logger = logging.getLogger(__name__)


def calculator_tool(expression: str) -> str:
    """
    Simple calculator tool that evaluates mathematical expressions.
    
    Args:
        expression: A mathematical expression as a string (e.g., "2 + 2", "10 * 5", "100 / 4")
    
    Returns:
        The result of the calculation as a string
    """
    print("🔧 Calculator tool has been called!")
    logger.info(f"🔧 Calculator tool called with expression: {expression}")
    
    try:
        # Use eval for simplicity - in production, consider using a safer parser
        # Only allow basic math operations for safety
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression. Only numbers and basic operators (+, -, *, /) are allowed."
        
        result = eval(expression)
        result_str = str(result)
        logger.info(f"🔧 Calculator result: {result_str}")
        return result_str
    except Exception as e:
        error_msg = f"Error calculating: {str(e)}"
        logger.error(f"🔧 {error_msg}")
        return error_msg


def get_calculator_tool_definition() -> Dict[str, Any]:
    """
    Get the tool definition for the calculator in OpenAI format.
    
    Returns:
        Tool definition dictionary compatible with OpenAI API
    """
    return {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "A simple calculator that can evaluate mathematical expressions. Use this tool when you need to perform calculations or arithmetic operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate (e.g., '2 + 2', '10 * 5', '100 / 4', '(5 + 3) * 2')"
                    }
                },
                "required": ["expression"]
            }
        }
    }


def execute_tool_call(tool_call: Dict[str, Any]) -> str:
    """
    Execute a tool call and return the result.
    
    Args:
        tool_call: Tool call dictionary from LLM response
    
    Returns:
        Result of tool execution as a string
    """
    function_name = tool_call.get("function", {}).get("name", "")
    function_args = tool_call.get("function", {}).get("arguments", "{}")
    
    try:
        args = json.loads(function_args) if isinstance(function_args, str) else function_args
    except json.JSONDecodeError:
        logger.error(f"Failed to parse tool arguments: {function_args}")
        return f"Error: Invalid arguments for {function_name}"
    
    if function_name == "calculator":
        expression = args.get("expression", "")
        return calculator_tool(expression)
    else:
        logger.warning(f"Unknown tool: {function_name}")
        return f"Error: Unknown tool {function_name}"


def _is_simple_calculation(query: str) -> bool:
    """
    Detect if the query is a simple mathematical calculation that doesn't need patient context.
    
    Args:
        query: User's input query
    
    Returns:
        True if the query appears to be a simple calculation
    """
    # Check for common calculation patterns
    calculation_patterns = [
        r'what is \d+',
        r'calculate \d+',
        r'compute \d+',
        r'\d+\s*[+\-*/]\s*\d+',  # Simple arithmetic like "10+10" or "5 * 3"
        r'what\'?s? \d+',
        r'how much is \d+',
    ]
    
    query_lower = query.lower().strip()
    
    # Check if it matches calculation patterns
    for pattern in calculation_patterns:
        if re.search(pattern, query_lower):
            # Make sure it's not asking about patient data (e.g., "what is the patient's age")
            if not any(word in query_lower for word in ['patient', 'age', 'weight', 'height', 'bmi', 'blood', 'pressure', 'lab', 'test', 'result']):
                return True
    
    return False


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

    # Skip patient context for simple calculations - let the calculator tool handle it
    if _is_simple_calculation(user_input):
        logger.info(f"🔧 Detected simple calculation, skipping patient context: {user_input}")
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
        # Add the current user input
        messages.append({"role": "user", "content": user_input})
        return messages, None

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

    # Debug logging: dump the exact context being sent to LLM
    try:
        final_user_message = messages[-1]["content"] if messages else ""
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open("demo.log", "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"CONTEXT DEBUG LOG - {timestamp_str}\n")
            f.write(f"Patient ID: {session_state.current_patient_id}\n")
            f.write(f"Timeline Mode: {timeline_mode}\n")
            f.write(f"User Query: {user_input}\n")
            f.write(f"Events Count: {len(final_events)}\n")
            f.write(f"Token Count: {sum(count_tokens(m['content']) for m in messages)}\n")
            f.write(f"{'='*80}\n")
            f.write("FINAL LLM CONTEXT:\n")
            f.write(f"{'='*80}\n")
            f.write(final_user_message)
            f.write(f"\n{'='*80}\n\n")
            
        logger.info(f"📝 Context dumped to demo.log ({len(final_user_message)} chars)")
    except Exception as e:
        logger.error(f"Error writing debug log: {e}")

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

    # Note: Model is specified per-request, not per-client
    # No need to switch clients when model changes

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

    # Define available tools
    tools = [get_calculator_tool_definition()]
    print(f"🔧 Tools registered: {[t.get('function', {}).get('name') for t in tools]}")
    logger.info(f"🔧 Tools registered: {[t.get('function', {}).get('name') for t in tools]}")

    try:
        # Generate response using secure-llm's native API with tools
        logger.info("🤖 Generating response...")
        print("🔧 Sending request with tools parameter")
        logger.info(f"🔧 Sending request with tools: {json.dumps(tools, indent=2)}")
        
        # Try to call with tools - some models/APIs might not support it
        try:
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice="auto",  # Let the model decide whether to use tools
            )
            print("🔧 API call with tools succeeded")
            logger.info("🔧 API call with tools succeeded")
        except (TypeError, ValueError) as e:
            # If tools parameter is not supported or secure-llm can't parse tool responses
            error_msg = str(e)
            if "Failed to parse OpenAI response" in error_msg or "NoneType" in error_msg:
                print(f"🔧 secure-llm cannot handle tool calls (expected): {error_msg}")
                print("🔧 Falling back to regular API call without tools")
                logger.warning(f"🔧 secure-llm cannot handle tool calls: {error_msg}")
            else:
                print(f"🔧 Tools parameter not supported: {e}")
                logger.warning(f"🔧 Tools parameter not supported by API: {e}")
            logger.warning("🔧 Falling back to regular API call without tools")
            # Fall back to regular call without tools
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        except Exception as e:
            print(f"🔧 Unexpected error calling API with tools: {e}")
            logger.error(f"🔧 Error calling API with tools: {e}")
            # Try fallback before giving up
            try:
                print("🔧 Attempting fallback to regular API call")
                response = llm_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
            except Exception as fallback_error:
                print(f"🔧 Fallback also failed: {fallback_error}")
                logger.error(f"🔧 Fallback also failed: {fallback_error}")
                raise

        # Debug: Log the response structure
        logger.debug(f"🔧 Response type: {type(response)}")
        if isinstance(response, dict):
            logger.debug(f"🔧 Response keys: {list(response.keys())}")
            if "choices" in response:
                logger.debug(f"🔧 Choices: {json.dumps(response.get('choices', [])[:1], indent=2, default=str)}")
        else:
            logger.debug(f"🔧 Response attributes: {dir(response)}")

        # Handle tool calls if present
        max_tool_iterations = 5  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_tool_iterations:
            # Extract response content and tool calls
            # Handle both dict and object-style responses
            if isinstance(response, dict):
                choices = response.get("choices", [])
            else:
                # Object-style response
                try:
                    choices = response.choices if hasattr(response, "choices") else []
                except AttributeError:
                    choices = []
            
            if not choices:
                logger.debug("🔧 No choices in response")
                break
            
            # Extract message from choice
            if isinstance(choices[0], dict):
                message = choices[0].get("message", {})
            else:
                message = choices[0].message if hasattr(choices[0], "message") else {}
            
            # Extract tool_calls
            if isinstance(message, dict):
                tool_calls = message.get("tool_calls")
                logger.debug(f"🔧 Message keys: {list(message.keys())}")
                logger.debug(f"🔧 Tool calls from dict: {tool_calls}")
            else:
                tool_calls = getattr(message, "tool_calls", None)
                logger.debug(f"🔧 Tool calls from object: {tool_calls}")
                # Try to access as attribute
                if tool_calls is None and hasattr(message, "__dict__"):
                    logger.debug(f"🔧 Message __dict__: {message.__dict__}")
            
            # If no tool calls, break and process the response normally
            if not tool_calls:
                print("🔧 No tool calls detected in response, processing normally")
                logger.info("🔧 No tool calls detected in response, processing normally")
                break
            
            # Convert tool_calls to list if needed
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls] if tool_calls else []
            
            # Extract message content
            if isinstance(message, dict):
                message_content = message.get("content")
            else:
                message_content = getattr(message, "content", None)
            
            # Add assistant message with tool calls to history
            assistant_message = {
                "role": "assistant",
                "content": message_content,
                "tool_calls": tool_calls
            }
            messages.append(assistant_message)
            
            # Execute all tool calls
            print(f"🔧 Executing {len(tool_calls)} tool call(s)")
            logger.info(f"🔧 Executing {len(tool_calls)} tool call(s)")
            for tool_call in tool_calls:
                # Convert tool_call to dict if it's an object
                if not isinstance(tool_call, dict):
                    tool_call_dict = {
                        "id": getattr(tool_call, "id", ""),
                        "function": {
                            "name": getattr(tool_call.function, "name", "") if hasattr(tool_call, "function") else "",
                            "arguments": getattr(tool_call.function, "arguments", "{}") if hasattr(tool_call, "function") else "{}"
                        }
                    }
                else:
                    tool_call_dict = tool_call
                
                logger.info(f"🔧 Executing tool: {tool_call_dict.get('function', {}).get('name')} with args: {tool_call_dict.get('function', {}).get('arguments')}")
                tool_result = execute_tool_call(tool_call_dict)
                tool_call_id = tool_call_dict.get("id", "")
                logger.info(f"🔧 Tool result: {tool_result}")
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_result
                })
            
            # Continue conversation with tool results
            iteration += 1
            logger.info(f"🔄 Tool call iteration {iteration}, continuing conversation...")
            
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice="auto",
            )
        
        # Extract final response content
        try:
            response_text = extract_response_content(response)
        except ValueError as e:
            logger.error(f"Error extracting response content: {e}")
            response_text = "[Error: Could not extract response content]"

        logger.info("✅ Response generated successfully")

        # Parse JSON response if present
        try:
            response_data = None
            
            # First, try to find JSON wrapped in code blocks
            json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                response_data = json.loads(json_str)
                logger.info("=== JSON Response from Code Block ===")
            else:
                # If no code block, try to find raw JSON object
                brace_positions = [i for i, char in enumerate(response_text) if char == '{']
                
                for start_pos in brace_positions:
                    json_candidate = response_text[start_pos:]
                    brace_count = 0
                    for end_pos, char in enumerate(json_candidate):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                try:
                                    potential_json = json_candidate[:end_pos + 1]
                                    response_data = json.loads(potential_json)
                                    logger.info("=== Raw JSON Response ===")
                                    break
                                except json.JSONDecodeError:
                                    continue
                    if response_data:
                        break

            if response_data:
                logger.info(json.dumps(response_data, indent=2))
                logger.info("=======================")

                # Store evidence data in session state for evidence panel
                if "evidence" in response_data and isinstance(response_data["evidence"], dict):
                    session_state.last_evidence_data = response_data["evidence"]
                    evidence_count = len(response_data["evidence"])
                    logger.info(f"📋 Stored evidence for {evidence_count} events in session_state")
                else:
                    session_state.last_evidence_data = {}
                    logger.info("📋 No evidence data found in response")

                # Use the answer from JSON, or full response if no answer key
                if "answer" in response_data:
                    history.append({"role": "assistant", "content": response_data["answer"]})
                else:
                    history.append({"role": "assistant", "content": response_text})
            else:
                # No JSON found, use raw response
                session_state.last_evidence_data = {}
                history.append({"role": "assistant", "content": response_text})

        except Exception as e:
            logger.error(f"Error parsing JSON response: {str(e)}")
            # Fallback to raw response
            session_state.last_evidence_data = {}
            history.append({"role": "assistant", "content": response_text})

        return history, history, fig

    except Exception as e:
        logger.error(f"Error during generation: {str(e)}", exc_info=True)
        history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        return history, history, fig
