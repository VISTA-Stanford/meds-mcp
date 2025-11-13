"""
UI Components for the MCP Chat Demo.
"""

import logging
import gradio as gr

logger = logging.getLogger(__name__)


class UIComponents:
    """Modular UI components for the demo."""

    @staticmethod
    def create_main_interface():
        """Create the main chat interface."""
        with gr.Accordion("Chat Interface", open=True):
            # Timeline visualization (initially hidden until data is loaded)
            timeline_plot = gr.Plot(value=None, show_label=False, visible=False)

            # Chat interface
            chatbot = gr.Chatbot(
                value=[], type="messages", elem_id="chatbot", show_copy_button=True
            )

            # Input row
            with gr.Row():
                with gr.Column(scale=1, min_width=200):
                    user_input = gr.Textbox(
                        label="Your message",
                        placeholder="Type a message and press enter",
                    )
                with gr.Column(scale=1, min_width=100):
                    datetime_input = gr.Textbox(
                        value="No data loaded",
                        interactive=True,
                        show_label=True,
                        label="Simulated Query Timestamp",
                    )

            # Example prompts
            gr.Examples(
                examples=[[prompt] for prompt in []],  # Will be filled by config
                inputs=user_input,
            )

            state = gr.State([])

            # Feedback handler
            def handle_feedback(x: gr.LikeData):
                logger.info(
                    f"Feedback: index={x.index}, value={x.value}, liked={x.liked}"
                )

            chatbot.like(handle_feedback, None, None, like_user_message=True)

        return timeline_plot, chatbot, user_input, datetime_input, state

    @staticmethod
    def create_chat_settings(defaults):
        """Create chat settings accordion."""
        with gr.Accordion("Chat Settings", open=False):
            with gr.Row():
                with gr.Column(scale=1, min_width=100):
                    use_timeline = gr.Checkbox(
                        label="Timeline Mode", value=defaults["use_timeline"]
                    )
                with gr.Column(scale=4, min_width=300):
                    max_input_length = gr.Slider(
                        minimum=1024,
                        maximum=256_000,
                        value=defaults["max_input_length"],
                        step=1024,
                        label="Max Input Length",
                    )

        return use_timeline, max_input_length

    @staticmethod
    def create_patient_loading():
        """Create patient loading panel."""
        with gr.Accordion("Load Patient Data", open=False):
            with gr.Row():
                patient_id_input = gr.Textbox(
                    label="Patient ID",
                    placeholder="Enter patient ID (e.g., 126692506)",
                    value="",
                    scale=3,
                )
                test_connection_btn = gr.Button("Test MCP Connection", scale=1)

            with gr.Row():
                load_btn = gr.Button("Load Patient", variant="primary")

            load_status = gr.Textbox(
                label="Status", value="No patient loaded", interactive=False
            )

        return patient_id_input, load_btn, load_status, test_connection_btn

    @staticmethod
    def create_llm_settings(defaults, available_models, default_model):
        """Create LLM engine settings accordion."""
        with gr.Accordion("LLM Engine Settings", open=False):
            with gr.Row():
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value=defaults["system_prompt"],
                    lines=6,
                    max_lines=15,
                )
            with gr.Row():
                prompt_template = gr.Textbox(
                    label="Prompt Template",
                    value=defaults["prompt_template"],
                    lines=8,
                    max_lines=20,
                    placeholder="Use {context} and {question} as placeholders",
                )
            with gr.Row():
                temperature = gr.Slider(
                    0, 1, value=defaults["temperature"], step=0.05, label="Temperature"
                )
                top_p = gr.Slider(
                    0, 1, value=defaults["top_p"], step=0.05, label="Top-p"
                )
                max_tokens = gr.Slider(
                    16,
                    8_192,
                    value=defaults["max_tokens"],
                    step=2048,
                    label="Max Output Tokens",
                )
            with gr.Row():
                model_selector = gr.Dropdown(
                    choices=[f"{model}" for model in available_models] if available_models else None,
                    value=default_model,
                    label="Model",
                    allow_custom_value=True,
                    info="Enter any model identifier (e.g., apim:gpt-4.1-mini, nero:gemini-2.0-flash)" if not available_models else None,
                )
                use_cache = gr.Checkbox(
                    label="Use Response Cache",
                    value=defaults["use_cache"],
                    info="Cache LLM responses to avoid redundant API calls",
                )

        return (
            system_prompt,
            prompt_template,
            temperature,
            top_p,
            max_tokens,
            model_selector,
            use_cache,
        )

    @staticmethod
    def update_examples(examples_component, example_prompts):
        """Update the examples component with the provided prompts."""
        examples_component.examples = [[prompt] for prompt in example_prompts]
