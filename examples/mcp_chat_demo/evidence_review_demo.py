"""
Evidence Review Demo - Production Integration

This integrates the real MCP chat functionality with our evidence review system.
Uses reliable button-based evidence links instead of problematic JavaScript bridges.

Key Features:
- Full MCP integration with real patient data
- Tabbed interface (Chat + Evidence Review)
- Evidence panel showing links from most recent message only
- Automatic tab switching when evidence is clicked
- Document viewer with highlighted evidence from MCP
- Validation workflow (Support/Reject evidence)
- Professional medical interface

Integration Status:
✅ Real chat interface from demo.py
✅ MCP patient data loading
✅ Evidence extraction from LLM JSON responses
✅ Evidence panel for most recent message
✅ Real evidence retrieval via MCP

python examples/mcp_chat_demo/evidence_review_demo.py \
--model "nero:gemini-2.5-pro" \
--mcp_url "http://localhost:8000/mcp" \
--patient_id 127672063


"""

import os
import sys
import logging
import re
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add current directory to path for local imports
sys.path.append(os.path.dirname(__file__))

import gradio as gr
import matplotlib

matplotlib.use("Agg")

# Import our modular components
from chat.config import parse_args, get_defaults, generate_system_prompt
from chat.core.session import session_state
from chat.ui.components import UIComponents
from chat.core.patient import load_patient_sync
from chat.visualization.timeline import update_query_datetime
from chat.llm.chat import stream_chat_response
from chat.mcp_client.client import test_connection_sync, get_event_by_id_sync
from chat.llm.secure_llm_client import get_llm_client, get_available_models
from chat.llm.cache import LLMCache
from chat.utils.vpn_check import check_vpn_combined

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global storage for user feedback on chat messages
chat_feedback = {}

# Global storage for evidence validation states
# Format: {message_index}_{event_id}: {"status": "validated/rejected/pending", "timestamp": "...", "message_index": int}
evidence_validations = {}


# Evidence retrieval functions are imported from chat.mcp_client.client
# Evidence extraction is now handled in chat.llm.chat.py and stored in session_state.last_evidence_data


def process_citations_in_response(response: str, evidence_data: Dict[str, List[str]]):
    """Replace [[event_id]] citations with styled HTML links."""
    # Pattern to match [[event_id]] citations
    citation_pattern = r'\[\[([^\]]+)\]\]'
    
    def replace_citation(match):
        event_id = match.group(1)
        # Create a styled link that matches the timeline blue theme
        return f'<a href="#" style="color: #1F78B4; text-decoration: underline; font-weight: 500;" onmouseover="this.style.color=\'#155A87\'" onmouseout="this.style.color=\'#1F78B4\'">[{event_id}]</a>'
    
    # Replace citations in the response text
    processed_response = re.sub(citation_pattern, replace_citation, response)
    
    # Get event IDs directly from evidence data keys (this is the correct source)
    event_ids = list(evidence_data.keys()) if evidence_data else []
    
    logger.info(f"🔗 Found {len(event_ids)} evidence IDs from JSON: {event_ids}")
    
    return processed_response, event_ids


def create_evidence_panel(event_ids: List[str], evidence_data: Dict[str, List[str]]):
    """Create evidence buttons for the given event IDs."""
    if not event_ids:
        return []
    
    buttons = []
    for event_id in event_ids:
        # Create a shortened display name
        display_name = event_id
        if len(event_id) > 20:
            display_name = f"{event_id[:10]}...{event_id[-7:]}"
        
        button_text = f"📋 {display_name}"
        buttons.append((button_text, event_id))
    
    return buttons


def generate_chat_download(history, current_evidence_data, current_event_ids):
    """Generate JSON file for chat history download."""
    try:
        # Get patient ID and create filename
        patient_id = session_state.current_patient_id or "unknown_patient"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Add microseconds to ensure uniqueness
        microseconds = datetime.now().strftime("%f")[:3]
        filename = f"{patient_id}_chat_{timestamp}_{microseconds}.json"
        
        # Collect all data
        download_data = {
            "metadata": {
                "patient_id": patient_id,
                "export_timestamp": datetime.now().isoformat(),
                "session_timestamp": session_state.query_datetime.isoformat() if session_state.query_datetime else None,
                "total_messages": len(history),
                "evidence_events": len(current_event_ids) if current_event_ids else 0
            },
            "chat_history": [],
            "user_feedback": chat_feedback.copy(),
            "evidence_data": current_evidence_data.copy() if current_evidence_data else {},
            "evidence_event_ids": current_event_ids.copy() if current_event_ids else [],
            "evidence_validations": evidence_validations.copy()
        }
        
        # Process chat history
        for i, message in enumerate(history):
            chat_entry = {
                "message_index": i,
                "role": message.get("role", "unknown"),
                "content": message.get("content", ""),
                "timestamp": datetime.now().isoformat(),  # Could be enhanced to track actual message times
                "has_user_feedback": i in chat_feedback,
                "evidence_validations": {}
            }
            
            # Add feedback if it exists for this message
            if i in chat_feedback:
                chat_entry["user_feedback"] = chat_feedback[i]
            
            # Add evidence validations for this message
            for validation_key, validation_data in evidence_validations.items():
                if validation_key.startswith(f"{i}_"):
                    event_id = validation_key.split("_", 1)[1]  # Extract event_id from validation_key
                    chat_entry["evidence_validations"][event_id] = validation_data
            
            download_data["chat_history"].append(chat_entry)
        
        # Create the JSON content
        json_content = json.dumps(download_data, indent=2, ensure_ascii=False)
        
        # Write to temporary file for download with the desired filename
        import tempfile
        import os
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, filename)
        
        # Ensure the file is properly written and closed
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(json_content)
            f.flush()  # Ensure content is written to disk
            os.fsync(f.fileno())  # Force write to disk
        
        logger.info(f"📥 Generated chat download: {filename} ({len(json_content)} chars)")
        logger.info(f"📋 Total evidence validations exported: {len(evidence_validations)}")
        return temp_path, filename
        
    except Exception as e:
        logger.error(f"Error generating chat download: {e}")
        # Return empty file on error
        import tempfile
        import os
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_filename = f"error_{timestamp}.json"
        temp_dir = tempfile.gettempdir()
        error_path = os.path.join(temp_dir, error_filename)
        
        with open(error_path, 'w', encoding='utf-8') as f:
            json.dump({"error": str(e)}, f)
        
        return error_path, error_filename


def create_demo():
    """Create the main Gradio demo interface with evidence review."""

    # Parse configuration
    args = parse_args()
    defaults = get_defaults()

    # Initialize LLM client and cache
    available_models = get_available_models()
    llm_client = get_llm_client(args.model)
    cache = LLMCache(cache_dir=args.cache_dir)

    logger.info(f"Initialized LLM client: {llm_client}, Model: {args.model}")

    with gr.Blocks(
        css="""
        .evidence-button {
            background: none !important;
            border: none !important;
            padding: 2px 4px !important;
            color: #007bff !important;
            text-decoration: underline !important;
            cursor: pointer !important;
            font-size: inherit !important;
            font-family: inherit !important;
            display: inline-block !important;
            margin: 2px !important;
        }
        .evidence-button:hover {
            background-color: #e3f2fd !important;
            border-radius: 3px !important;
        }
        .evidence-button:focus {
            outline: 2px solid #007bff !important;
            outline-offset: 2px !important;
        }
        """
    ) as demo:
        
        # Global state for evidence
        current_evidence_data = gr.State({})
        current_event_ids = gr.State([])
        
        with gr.Tabs() as tabs:
            with gr.Tab("Chat", id=0):
                # Create main interface components individually for custom layout
                with gr.Accordion("Chat Interface", open=True):
                    # Timeline visualization (initially hidden until data is loaded)
                    timeline_plot = gr.Plot(value=None, show_label=False, visible=False)

                    # Chat interface
                    chatbot = gr.Chatbot(
                        value=[], type="messages", elem_id="chatbot", show_copy_button=True
                    )

                    # Evidence Panel - directly below chatbot
                    with gr.Accordion("Evidence References", open=True):
                        #gr.Markdown("*Evidence links will appear here when the LLM provides citations.*")
                        
                        # Horizontal row with wrapping for evidence buttons
                        with gr.Row(elem_id="evidence_btn_row"):
                            evidence_buttons_container = gr.Column(visible=False)
                            
                            # Dynamic evidence buttons (initially empty)
                            evidence_buttons = {}
                            for i in range(10):  # Pre-create up to 10 evidence buttons
                                evidence_buttons[f"btn_{i}"] = gr.Button(
                                    f"Evidence {i}", 
                                    visible=False,
                                    elem_classes=["evidence-button-horizontal"],
                                    elem_id=f"evidence_btn_{i}",
                                    size="sm",
                                    scale=1
                                )

                    # Input row at bottom
                    with gr.Row():
                        with gr.Column(scale=3, min_width=200):
                            user_input = gr.Textbox(
                                label="Your message",
                                placeholder="Type a message and press enter",
                            )
                        with gr.Column(scale=2, min_width=100):
                            datetime_input = gr.Textbox(
                                value="No data loaded",
                                interactive=True,
                                show_label=True,
                                label="Simulated Query Timestamp",
                            )
                        with gr.Column(scale=1, min_width=80):
                            download_btn = gr.DownloadButton(
                                label="💾 Download Chat",
                                value=None,
                                size="sm",
                                variant="secondary"
                            )

                    # Add example prompts
                    gr.Examples(
                        examples=[[prompt] for prompt in defaults["example_prompts"]],
                        inputs=user_input,
                    )

                    state = gr.State([])

                    # Feedback handler for chatbot
                    def handle_feedback(x: gr.LikeData):
                        logger.info(
                            f"Feedback: index={x.index}, value={x.value}, liked={x.liked}"
                        )
                        # Store feedback globally for download
                        chat_feedback[x.index] = {
                            "liked": x.liked,
                            "value": x.value,
                            "timestamp": datetime.now().isoformat()
                        }

                    chatbot.like(handle_feedback, None, None, like_user_message=True)

                # Other UI components
                use_timeline, max_input_length = UIComponents.create_chat_settings(defaults)
                patient_id_input, load_btn, load_status, test_connection_btn = (
                    UIComponents.create_patient_loading()
                )
                (
                    system_prompt,
                    prompt_template,
                    temperature,
                    top_p,
                    max_tokens,
                    model_selector,
                    use_cache,
                ) = UIComponents.create_llm_settings(defaults, available_models, args.model)

            with gr.Tab("Evidence Review", id=1):
                gr.Markdown("## Evidence Review & Validation")
                gr.Markdown("Review source documents and validate evidence quality.")
                
                # Evidence metadata
                with gr.Row():
                    event_id_display = gr.Textbox(
                        label="Event ID",
                        value="No evidence selected",
                        interactive=False,
                        scale=2
                    )
                    evidence_status = gr.Textbox(
                        label="Validation Status",
                        value="Pending Review",
                        interactive=False,
                        scale=1
                    )
                
                # Document viewer
                document_viewer = gr.HTML(
                    value="""
                    <div style='padding: 30px; border: 2px dashed #ccc; border-radius: 10px; text-align: center; color: #666;'>
                        <h3>No Evidence Selected</h3>
                        <p>Click an evidence link in the chat to load the source document here.</p>
                        <p>You'll be able to review the full context and validate evidence quality.</p>
                    </div>
                    """,
                    label="Source Document"
                )
                
                # Evidence snippets (would show highlighted portions)
                evidence_snippets = gr.JSON(
                    label="Evidence Snippets",
                    value={},
                    visible=False
                )
                
                # Hidden state to track current validation key
                current_validation_key = gr.State("")
                
                # Validation buttons
                with gr.Row():
                    validate_yes_btn = gr.Button("✅ Evidence Supports Claim", variant="primary", scale=1)
                    validate_no_btn = gr.Button("❌ Evidence Does Not Support", variant="stop", scale=1)
                    back_to_chat_btn = gr.Button("← Back to Chat", variant="secondary", scale=1)

        # Event handlers from demo.py
        def load_patient_wrapper(patient_id):
            result = load_patient_sync(patient_id, args.mcp_url)
            if len(result) >= 5:
                patient_id, message, fig, datetime_str, timeline_visible = result
                new_system_prompt = generate_system_prompt(datetime_str)
                logger.info(f"🤖 Updated system prompt for loaded patient with date: {datetime_str}")
                logger.info(f"📊 Timeline visibility: {timeline_visible}")
                return patient_id, message, gr.update(value=fig, visible=timeline_visible), datetime_str, new_system_prompt
            else:
                return result + (generate_system_prompt(), gr.update(visible=False))

        def test_connection_wrapper():
            return test_connection_sync(args.mcp_url)

        def chat_wrapper(*chat_args):
            # Get the chat response
            history, state_new, fig = stream_chat_response(*chat_args, args.mcp_url, llm_client)
            
            # Get evidence data from session_state (now properly stored by chat.py)
            evidence_data = session_state.last_evidence_data or {}
            event_ids = list(evidence_data.keys()) if evidence_data else []
            
            logger.info(f"🔗 Evidence data from session_state: {len(evidence_data)} events")
            if evidence_data:
                logger.info(f"   Event IDs: {list(evidence_data.keys())}")
            
            # Process citations in the most recent assistant message (optional, for display)
            if history and len(history) > 0:
                last_message = history[-1]
                if last_message.get("role") == "assistant":
                    # Process citations for display (replace [[id]] with [id])
                    processed_response, _ = process_citations_in_response(
                        last_message["content"], evidence_data
                    )
                    
                    # Update the response with processed citations
                    history[-1]["content"] = processed_response
            
            # Update evidence buttons visibility
            button_updates = []
            for i in range(10):
                if i < len(event_ids):
                    event_id = event_ids[i]
                    display_name = event_id
                    if len(event_id) > 25:
                        display_name = f"{event_id[:12]}...{event_id[-10:]}"
                    button_updates.append(gr.update(value=f"📋 {display_name}", visible=True))
                else:
                    button_updates.append(gr.update(visible=False))
            
            # Show evidence container if we have evidence
            container_update = gr.update(visible=len(event_ids) > 0)
            
            return [history, state_new, fig, evidence_data, event_ids, container_update] + button_updates

        def handle_clear():
            """Handle the clear event when user clicks the trash icon."""
            logger.info("🗑️ Chat history cleared by user")
            
            # Clear feedback and validation storage
            global chat_feedback, evidence_validations
            chat_feedback.clear()
            evidence_validations.clear()
            
            # Clear evidence buttons
            button_updates = [gr.update(visible=False) for _ in range(10)]
            container_update = gr.update(visible=False)
            
            return [[], [], {}, [], container_update] + button_updates

        def handle_download(history, current_evidence_data, current_event_ids):
            """Handle chat history download."""
            logger.info("📥 User requested chat history download")
            
            if not history:
                logger.warning("No chat history to download")
                return None
                
            try:
                temp_path, filename = generate_chat_download(history, current_evidence_data, current_event_ids)
                
                # Ensure file exists and is readable
                import os
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                    logger.info(f"📁 File ready for download: {temp_path} ({os.path.getsize(temp_path)} bytes)")
                    # Return the file path - no gr.update() wrapper
                    return temp_path
                else:
                    logger.error(f"❌ Generated file not found or empty: {temp_path}")
                    return None
                    
            except Exception as e:
                logger.error(f"❌ Error in download handler: {e}")
                return None

        def update_datetime_and_system_prompt(datetime_str: str):
            """Update both the query datetime and system prompt when datetime changes."""
            try:
                updated_datetime, fig, timeline_visible = update_query_datetime(datetime_str)
                new_system_prompt = generate_system_prompt(updated_datetime)
                logger.info(f"📅 Updated datetime to: {updated_datetime}")
                logger.info(f"🤖 Updated system prompt with date: {updated_datetime}")
                logger.info(f"📊 Timeline visibility: {timeline_visible}")
                return updated_datetime, gr.update(value=fig, visible=timeline_visible), new_system_prompt
            except Exception as e:
                logger.error(f"Error updating datetime and system prompt: {e}")
                current_str = (
                    session_state.query_datetime.strftime("%Y-%m-%d %H:%M:%S")
                    if session_state.query_datetime
                    else "No data loaded"
                )
                return current_str, gr.update(visible=False), generate_system_prompt(current_str)

        def load_evidence(event_id: str, evidence_data: dict, current_history):
            """Load evidence for review using MCP."""
            logger.info(f"🔍 Loading evidence: {event_id}")
            
            # Get current message index (evidence is from most recent message)
            message_index = len(current_history) - 1 if current_history else 0
            validation_key = f"{message_index}_{event_id}"
            
            # Check for existing validation state
            existing_validation = evidence_validations.get(validation_key, {})
            initial_status = existing_validation.get("status", "Pending Review")
            
            logger.info(f"📋 Evidence validation key: {validation_key}, existing status: {initial_status}")
            
            try:
                # Get event from MCP server
                success, event_data, error = get_event_by_id_sync(event_id, args.mcp_url)
                
                if success and event_data:
                    # Format document content
                    content = event_data.get("content", "No content available")
                    metadata = event_data.get("metadata", {})
                    
                    # Get evidence snippets for highlighting
                    snippets = evidence_data.get(event_id, [])
                    
                    # Create highlighted document with proper newline handling
                    highlighted_content = content
                    highlight_count = 0
                    
                    for snippet in snippets:
                        # Normalize the snippet for matching (handle different newline representations)
                        normalized_snippet = snippet.replace('\\n', '\n').strip()
                        
                        # Try exact match first
                        if normalized_snippet in highlighted_content:
                            highlighted_content = highlighted_content.replace(
                                normalized_snippet,
                                f"<mark style='background-color: #ffeb3b; padding: 2px;'>{normalized_snippet}</mark>",
                                1  # Replace only first occurrence
                            )
                            highlight_count += 1
                            logger.info(f"   ✅ Exact match highlighted: '{normalized_snippet[:50]}...'")
                        else:
                            # Try fuzzy matching with flexible whitespace
                            import re
                            
                            # Split snippet into words and create flexible pattern
                            words = re.split(r'\s+', normalized_snippet.strip())
                            if len(words) > 1:
                                # Create pattern: word1 + flexible whitespace + word2 + ...
                                pattern_parts = []
                                for i, word in enumerate(words):
                                    pattern_parts.append(re.escape(word))
                                    if i < len(words) - 1:  # Not the last word
                                        pattern_parts.append(r'\s+')  # Flexible whitespace
                                
                                pattern = ''.join(pattern_parts)
                                
                                try:
                                    match = re.search(pattern, highlighted_content, re.DOTALL)
                                    if match:
                                        matched_text = match.group(0)
                                        highlighted_content = highlighted_content.replace(
                                            matched_text,
                                            f"<mark style='background-color: #ffeb3b; padding: 2px;'>{matched_text}</mark>",
                                            1  # Replace only first occurrence
                                        )
                                        highlight_count += 1
                                        logger.info(f"   ✅ Fuzzy match highlighted: '{matched_text[:50]}...'")
                                    else:
                                        logger.warning(f"   ❌ No match for snippet: '{normalized_snippet[:50]}...'")
                                except re.error as e:
                                    logger.warning(f"   ❌ Regex error for snippet: {str(e)}")
                            else:
                                # Single word, try case-insensitive match
                                word = re.escape(words[0])
                                try:
                                    match = re.search(word, highlighted_content, re.IGNORECASE)
                                    if match:
                                        matched_text = match.group(0)
                                        highlighted_content = highlighted_content.replace(
                                            matched_text,
                                            f"<mark style='background-color: #ffeb3b; padding: 2px;'>{matched_text}</mark>",
                                            1
                                        )
                                        highlight_count += 1
                                        logger.info(f"   ✅ Word match highlighted: '{matched_text}'")
                                except re.error:
                                    logger.warning(f"   ❌ Word match failed: '{words[0]}'")
                    
                    logger.info(f"📋 Highlighted {highlight_count}/{len(snippets)} evidence snippets")
                    
                    document_html = f"""
                    <div style='padding: 20px; border: 1px solid #ddd; border-radius: 8px; background: #f9f9f9;'>
                        <h4>Medical Record - Event {event_id}</h4>
                        <p><strong>Event ID:</strong> {event_id}</p>
                        {f"<p><strong>Date:</strong> {metadata.get('timestamp', 'Unknown')}</p>" if metadata.get('timestamp') else ""}
                        <hr>
                        <div style='background: white; padding: 15px; border-radius: 5px; font-family: monospace; white-space: pre-wrap;'>
                            {highlighted_content}
                        </div>
                    </div>
                    """
                    
                    logger.info(f"✅ Evidence loaded, switching to Evidence Review tab")
                    
                    return (
                        gr.update(selected=1),           # Switch to Evidence Review tab
                        event_id,                       # Update event ID display
                        initial_status,                 # Update validation status (restored from memory)
                        document_html,                  # Update document viewer
                        {event_id: snippets},          # Update evidence snippets
                        validation_key                  # Pass validation key for button handlers
                    )
                else:
                    error_msg = error or "Event not found"
                    logger.error(f"❌ Failed to load evidence: {error_msg}")
                    
                    return (
                        gr.update(selected=1),
                        event_id,
                        "Error",
                        f"""
                        <div style='padding: 20px; border: 1px solid #f44336; border-radius: 8px; background: #ffebee;'>
                            <h4>Error Loading Evidence</h4>
                            <p><strong>Event ID:</strong> {event_id}</p>
                            <p><strong>Error:</strong> {error_msg}</p>
                        </div>
                        """,
                        {},
                        validation_key
                    )
                    
            except Exception as e:
                logger.error(f"❌ Exception loading evidence: {e}")
                return (
                    gr.update(selected=1),
                    event_id,
                    "Error",
                    f"""
                    <div style='padding: 20px; border: 1px solid #f44336; border-radius: 8px; background: #ffebee;'>
                        <h4>Exception Loading Evidence</h4>
                        <p><strong>Event ID:</strong> {event_id}</p>
                        <p><strong>Exception:</strong> {str(e)}</p>
                    </div>
                    """,
                    {},
                    validation_key
                )

        def validate_evidence_positive(validation_key):
            """Mark evidence as supporting the claim."""
            logger.info(f"✅ Evidence validated as SUPPORTING: {validation_key}")
            
            # Save validation state persistently
            global evidence_validations
            evidence_validations[validation_key] = {
                "status": "✅ VALIDATED - Supports Claim",
                "decision": "supports",
                "timestamp": datetime.now().isoformat()
            }
            
            return "✅ VALIDATED - Supports Claim"

        def validate_evidence_negative(validation_key):
            """Mark evidence as not supporting the claim."""
            logger.info(f"❌ Evidence validated as NOT SUPPORTING: {validation_key}")
            
            # Save validation state persistently
            global evidence_validations
            evidence_validations[validation_key] = {
                "status": "❌ REJECTED - Does Not Support Claim",
                "decision": "rejects",
                "timestamp": datetime.now().isoformat()
            }
            
            return "❌ REJECTED - Does Not Support Claim"

        def switch_to_chat():
            """Switch back to chat tab."""
            logger.info("🔄 Returning to chat")
            return gr.update(selected=0)

        # Wire up event handlers
        load_btn.click(
            load_patient_wrapper,
            inputs=[patient_id_input],
            outputs=[patient_id_input, load_status, timeline_plot, datetime_input, system_prompt],
        )

        test_connection_btn.click(
            test_connection_wrapper,
            inputs=[],
            outputs=[load_status],
        )

        # Chat input handler with evidence processing
        chat_outputs = [
            chatbot, state, timeline_plot, current_evidence_data, current_event_ids, evidence_buttons_container
        ] + [evidence_buttons[f"btn_{i}"] for i in range(10)]

        user_input.submit(
            chat_wrapper,
            inputs=[
                user_input,
                state,
                system_prompt,
                temperature,
                top_p,
                max_tokens,
                use_timeline,
                model_selector,
                max_input_length,
                prompt_template,
                use_cache,
            ],
            outputs=chat_outputs,
        )

        datetime_input.submit(
            update_datetime_and_system_prompt,
            inputs=[datetime_input],
            outputs=[datetime_input, timeline_plot, system_prompt],
        )

        # Clear event handler with evidence cleanup
        clear_outputs = [
            chatbot, state, current_evidence_data, current_event_ids, evidence_buttons_container
        ] + [evidence_buttons[f"btn_{i}"] for i in range(10)]

        chatbot.clear(
            handle_clear,
            inputs=[],
            outputs=clear_outputs,
        )

        # Download button handler  
        download_btn.click(
            handle_download,
            inputs=[chatbot, current_evidence_data, current_event_ids],
            outputs=download_btn
        )

        # Wire up evidence buttons
        for i in range(10):
            def make_evidence_handler(button_index):
                def evidence_handler(evidence_data, event_ids, current_history):
                    if button_index < len(event_ids):
                        event_id = event_ids[button_index]
                        return load_evidence(event_id, evidence_data, current_history)
                    return gr.update(), "", "", "", {}, ""
                return evidence_handler

            evidence_buttons[f"btn_{i}"].click(
                make_evidence_handler(i),
                inputs=[current_evidence_data, current_event_ids, chatbot],
                outputs=[tabs, event_id_display, evidence_status, document_viewer, evidence_snippets, current_validation_key]
            )

        # Validation buttons
        validate_yes_btn.click(
            validate_evidence_positive,
            inputs=[current_validation_key],
            outputs=[evidence_status]
        )

        validate_no_btn.click(
            validate_evidence_negative,
            inputs=[current_validation_key],
            outputs=[evidence_status]
        )

        back_to_chat_btn.click(
            switch_to_chat,
            inputs=[],
            outputs=[tabs]
        )

        # Auto-load patient if provided via command line
        def auto_load_patient():
            """Auto-load patient on startup if patient_id is provided."""
            if args.patient_id:
                logger.info(f"🚀 Auto-loading patient {args.patient_id}")
                try:
                    result = load_patient_wrapper(args.patient_id)
                    return result
                except Exception as e:
                    logger.error(f"Error auto-loading patient: {e}")
                    return args.patient_id, f"Auto-load failed: {str(e)}", gr.update(visible=False), "No data loaded", generate_system_prompt()
            else:
                return "", "No patient specified for auto-load", gr.update(visible=False), "No data loaded", generate_system_prompt()

        # Wire up auto-load on interface load
        demo.load(
            auto_load_patient,
            inputs=[],
            outputs=[patient_id_input, load_status, timeline_plot, datetime_input, system_prompt],
        )

    return demo


if __name__ == "__main__":
    # Check VPN connectivity before launching
    print("=" * 60)
    print("Checking Stanford VPN connectivity...")
    print("=" * 60)
    
    vpn_connected, vpn_message = check_vpn_combined()
    print(vpn_message)
    print()
    
    if not vpn_connected:
        print("⚠️  WARNING: Not connected to Stanford VPN")
        print("   This application requires VPN access to Stanford services.")
        print("   Some features may not work correctly without VPN.")
        print()
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("Exiting. Please connect to Stanford VPN and try again.")
            sys.exit(1)
        print()
    
    # Create and launch the demo
    demo = create_demo()
    
    # Add custom CSS for horizontal wrapping evidence buttons
    demo.load(js="""
    function() {
        // Add custom CSS for evidence buttons and row layout
        const style = document.createElement('style');
        style.textContent = `
            /* Evidence buttons row with flex-wrap */
            #evidence_btn_row {
                display: flex !important;
                flex-wrap: wrap !important;
                gap: 0.5rem !important;
                align-items: flex-start !important;
                justify-content: flex-start !important;
            }
            
            /* Individual evidence button styling */
            .evidence-button-horizontal {
                background: linear-gradient(45deg, #1F78B4, #3A9BC1) !important;
                color: white !important;
                border: none !important;
                text-decoration: none !important;
                cursor: pointer !important;
                margin: 2px !important;
                padding: 8px 16px !important;
                border-radius: 6px !important;
                font-size: 13px !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
                transition: all 0.2s ease !important;
                text-align: center !important;
                font-weight: 500 !important;
                white-space: nowrap !important;
                min-width: 140px !important;
                max-width: 140px !important;
                flex-shrink: 0 !important;
               
            }
            .evidence-button-horizontal:hover {
                background: linear-gradient(45deg, #155A87, #2E7A9B) !important;
                transform: translateY(-1px) !important;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
            }
            .evidence-button-horizontal:active {
                transform: translateY(0px) !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.15) !important;
            }
        `;
        document.head.appendChild(style);
    }
    """)

    demo.launch(share=False, inbrowser=True) 