"""
Refactored Patient Chat Demo using MCP Server

Usage:

python apps/mcp_chat_demo/demo.py \
--model "nero:gemini-2.0-flash" \
--mcp_url "http://localhost:8000/mcp"


python examples/mcp_chat_demo/demo.py \
--model "nero:gemini-2.5-pro" \
--mcp_url "http://localhost:8000/mcp" \
--patient_id 127672063


This demo uses an MCP server for patient data access instead of local XML files.
The user provides a patient ID and all queries are anchored to that patient.
"""

import os
import sys
import logging

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
from chat.mcp_client.client import test_connection_sync
from chat.llm.secure_llm_client import get_llm_client, get_available_models
from chat.llm.cache import LLMCache

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_demo():
    """Create the main Gradio demo interface."""

    # Parse configuration
    args = parse_args()
    defaults = get_defaults()

    # Initialize LLM client and cache
    available_models = get_available_models()
    llm_client = get_llm_client(args.model)
    cache = LLMCache(cache_dir=args.cache_dir)

    logger.info(f"Initialized LLM client: {llm_client}, Model: {args.model}")

    with gr.Blocks() as demo:
        # Create all UI components
        timeline_plot, chatbot, user_input, datetime_input, state = (
            UIComponents.create_main_interface()
        )
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

        # Add example prompts
        gr.Examples(
            examples=[[prompt] for prompt in defaults["example_prompts"]],
            inputs=user_input,
        )

        # Event handlers - wrap functions to include required parameters
        def load_patient_wrapper(patient_id):
            result = load_patient_sync(patient_id, args.mcp_url)
            # result is: (patient_id, message, fig, datetime_str, timeline_visible, success)
            if len(result) >= 6:
                patient_id, message, fig, datetime_str, timeline_visible, success = result
                # Generate updated system prompt with the new datetime
                new_system_prompt = generate_system_prompt(datetime_str)
                logger.info(f"🤖 Updated system prompt for loaded patient with date: {datetime_str}")
                logger.info(f"📊 Timeline visibility: {timeline_visible}")
                return patient_id, message, gr.update(value=fig, visible=timeline_visible), datetime_str, new_system_prompt
            else:
                # Fallback for error cases
                return result + (generate_system_prompt(), gr.update(visible=False))

        def test_connection_wrapper():
            return test_connection_sync(args.mcp_url)

        def chat_wrapper(*chat_args):
            # Add mcp_url and llm_client to the arguments
            return stream_chat_response(*chat_args, args.mcp_url, llm_client)

        def handle_clear():
            """Handle the clear event when user clicks the trash icon."""
            logger.info("🗑️ Chat history cleared by user")
            return [], []  # Clear both chatbot display and state

        def update_datetime_and_system_prompt(datetime_str: str):
            """Update both the query datetime and system prompt when datetime changes."""
            try:
                # Update the datetime (using existing function)
                updated_datetime, fig, timeline_visible = update_query_datetime(datetime_str)
                
                # Generate new system prompt with the updated date
                new_system_prompt = generate_system_prompt(updated_datetime)
                
                logger.info(f"📅 Updated datetime to: {updated_datetime}")
                logger.info(f"🤖 Updated system prompt with date: {updated_datetime}")
                logger.info(f"📊 Timeline visibility: {timeline_visible}")
                
                return updated_datetime, gr.update(value=fig, visible=timeline_visible), new_system_prompt
                
            except Exception as e:
                logger.error(f"Error updating datetime and system prompt: {e}")
                # Return current values on error
                current_str = (
                    session_state.query_datetime.strftime("%Y-%m-%d %H:%M:%S")
                    if session_state.query_datetime
                    else "No data loaded"
                )
                return current_str, gr.update(visible=False), generate_system_prompt(current_str)

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
            outputs=[chatbot, state, timeline_plot],
        )

        datetime_input.submit(
            update_datetime_and_system_prompt,
            inputs=[datetime_input],
            outputs=[datetime_input, timeline_plot, system_prompt],
        )

        # Clear event handler
        chatbot.clear(
            handle_clear,
            inputs=[],
            outputs=[chatbot, state],
        )

        # Auto-load patient if provided via command line
        def auto_load_patient():
            """Auto-load patient on startup if patient_id is provided."""
            if args.patient_id:
                logger.info(f"🚀 Auto-loading patient {args.patient_id}")
                try:
                    # Use the same wrapper function as the button click
                    result = load_patient_wrapper(args.patient_id)
                    # result is: (patient_id, message, timeline_update, datetime_str, system_prompt)
                    return result
                except Exception as e:
                    logger.error(f"Error auto-loading patient: {e}")
                    return args.patient_id, f"Auto-load failed: {str(e)}", gr.update(visible=False), "No data loaded", generate_system_prompt()
            else:
                # No patient ID provided, return empty state
                return "", "No patient specified for auto-load", gr.update(visible=False), "No data loaded", generate_system_prompt()

        # Wire up auto-load on interface load
        demo.load(
            auto_load_patient,
            inputs=[],
            outputs=[patient_id_input, load_status, timeline_plot, datetime_input, system_prompt],
        )

    return demo


if __name__ == "__main__":
    # Create and launch the demo
    demo = create_demo()
    demo.launch(share=False, inbrowser=True)
