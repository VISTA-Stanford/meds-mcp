"""
Refactored Patient Chat Demo using MCP Server

Usage:

python apps/mcp_chat_demo/demo.py \
--model "nero:gemini-2.0-flash" \
--mcp_url "http://localhost:8000/mcp"


python apps/mcp_chat_demo/demo.py \
--model "apim:gpt-4o-mini" \
--mcp_url "http://localhost:8000/mcp"

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
from chat.config import parse_args, get_defaults
from chat.core.session import session_state
from chat.ui.components import UIComponents
from chat.core.patient import load_patient_sync
from chat.visualization.timeline import update_query_datetime
from chat.llm.chat import stream_chat_response
from chat.mcp_client.client import test_connection_sync
from lumia.engines import get_llm_client, get_available_models
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
            return load_patient_sync(patient_id, args.mcp_url)

        def test_connection_wrapper():
            return test_connection_sync(args.mcp_url)

        def chat_wrapper(*chat_args):
            # Add mcp_url and llm_client to the arguments
            return stream_chat_response(*chat_args, args.mcp_url, llm_client)

        def handle_clear():
            """Handle the clear event when user clicks the trash icon."""
            logger.info("🗑️ Chat history cleared by user")
            return [], []  # Clear both chatbot display and state

        # Wire up event handlers
        load_btn.click(
            load_patient_wrapper,
            inputs=[patient_id_input],
            outputs=[patient_id_input, load_status, timeline_plot, datetime_input],
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
            update_query_datetime,
            inputs=[datetime_input],
            outputs=[datetime_input, timeline_plot],
        )

        # Clear event handler
        chatbot.clear(
            handle_clear,
            inputs=[],
            outputs=[chatbot, state],
        )

        # Auto-load patient if provided via command line
        if args.patient_id:
            logger.info(f"🚀 Auto-loading patient {args.patient_id}")
            try:
                # Set initial patient ID
                patient_id_input.value = args.patient_id
                # Auto-load the patient
                initial_result = load_patient_sync(args.patient_id, args.mcp_url)
                if len(initial_result) >= 2:
                    load_status.value = initial_result[1]
                    if len(initial_result) >= 4:
                        datetime_input.value = initial_result[3]
                    if len(initial_result) >= 3 and initial_result[2] is not None:
                        timeline_plot.value = initial_result[2]
            except Exception as e:
                logger.error(f"Error auto-loading patient: {e}")
                load_status.value = f"Auto-load failed: {str(e)}"

    return demo


if __name__ == "__main__":
    # Create and launch the demo
    demo = create_demo()
    demo.launch(share=False, inbrowser=True)
