"""
Configuration and settings for the MCP Chat Demo.
"""

import argparse
from datetime import date
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ========== Parse command line arguments ==========
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Patient Chat with MCP Server")
    parser.add_argument(
        "--model",
        default="apim:gpt-4o-mini",
        help="Default LLM model to use (e.g., 'apim:gpt-4o-mini', 'apim:claude-3.5', etc.)",
    )
    parser.add_argument(
        "--cache_dir",
        default=".cache",
        help="Directory to store the LLM response cache",
    )
    parser.add_argument(
        "--mcp_url", default="http://localhost:8000/mcp", help="MCP server URL"
    )
    parser.add_argument(
        "--patient_id", default="127672063", help="Patient ID to auto-load on startup"
    )
    return parser.parse_args()


# ========== Default Settings ==========
def get_defaults():
    """Get default configuration settings."""
    todays_date_str = date.today().strftime("%B %d, %Y")

    return {
        "model": "nero:gemini-2.0-flash",
        "system_prompt": f"""You are a helpful EHR assistant. **Today's date is {todays_date_str}**.

You are given a patient's EHR data and a question. Your task is to answer the question **based solely on the provided EHR data**.""",
        "prompt_template": """You are provided with relevant patient information below:

{context}

Using this information, complete the following instruction:

{question}

Respond with a **JSON object** in the following format:

```json
{{
  "answer": "<short answer formatted in markdown, with inline uid citations>",
  "evidence": {{
    "uid1": ["<verbatim text snippet of supporting evidence for uid1>", ...],
    "uid2": ["<verbatim text snippet of supporting evidence for uid2>", ...],
    ...
  }}
}}
```

* The `answer` field should provide a concise response to the question. Include relevent dates and times. The text desciption should be an brief summary. Format everything in markdown.
* Include a reference to motivate each statement with the corresponding `uid` from the evidence list using inline references as [[uid1]], [[uid2]], etc.
* The `evidence` field should list only the `uid`s of `<event>` tags that directly support your answer.
* Text supporting the answer should be a verabtim substring copy of the source context XML. Include the minimal substring that supports the answer.

**Only include essential evidence. Avoid citing unrelated or extraneous events.**""",
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 8_192,
        "max_input_length": 8_192,
        "use_timeline": False,
        "use_cache": False,
        "use_streaming": True,
        "example_prompts": [
            "Calculate 10-year cardiovascular risk using the most recent LDL, blood pressure, and smoking status.",
            "List all <event> tags of type 'image' with their corresponding imaging modality (e.g., CT, MRI, etc.), list in chronological order, include the date and anatomical location.",
            "List all medical imaging events (e.g., CT, MRI, etc.) in chronological order, include the modality (in bold), date of scan, and anatomical location.",
            "List inpatient admissions or ED visits in the past five years, with date, reason, and length of stay.",
            "Extract mentions of housing instability, food insecurity, or transportation issues and their dates.",
            "List current medications and any dose changes over the past six months.",
            "Outline a summary of this patient's cancer diagnoses, treatments, and corresponding responses. Include specific dates and evidence from the EHR.",
        ],
    }
