"""
Configuration and settings for the MCP Chat Demo.
"""

import argparse
from datetime import date
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

TUMOR_BOARD_PROMPT = """TASK: You are assisting a cancer tumor board. Using the most recent progress notes and the tumor board discussion question, generate a concise case summary.

TUMOR BOARD QUESTION:
Do we pursue neoadjuvant chemotherapy then re-attempt surgery, or move forward with radiation instead?


OUTPUT (≤999 characters including spaces). Use ONLY this format and numeric dates when relevant:
AIGen: [LASTNAME]: [AGE] [GENDER] with h/o [CANCER_TYPE + key pathology details]
Prior therapy: [THERapy details in chronological order]
Tumor board question: [QUESTION]

TUMOR-SPECIFIC ADDITIONS (append within the cancer/type clause when available):
- Lung: histology (e.g., adenocarcinoma/squamous), smoking hx, mutations (e.g., KRAS G12C, EGFR Ex19 del), PD-L1 (e.g., TPS=5%)
- Thymoma/thymic carcinoma: WHO subtype, Masaoka stage (no mutations/PD-L1)
- Neuroendocrine: Ki-67, DOTATATE scan
- Mesothelioma: subtype (epithelioid/biphasic/sarcomatoid)
- Pancreatic: MSI, MMR, HER2, KRAS

RULES:
- Abbreviations OK (e.g., NSCLC).
- No extra commentary before/after the formatted output.
- Only include treatments for the primary cancer under discussion; ignore prior unrelated cancers.
- Summarize only the most recent imaging.
- Omit fields if data is absent—do not fabricate.
"""


# ========== Parse command line arguments ==========
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Patient Chat with MCP Server")
    
    available_models = [
        'apim:gpt-4.1', 'apim:gpt-4.1-mini', 'apim:gpt-4.1-nano', 'apim:o3-mini',
        'apim:claude-3.5', 'apim:claude-3.7', 'apim:gemini-2.0-flash',
        'apim:gemini-2.5-pro-preview-05-06', 'apim:llama-3.3-70b',
        'apim:llama-4-maverick-17b', 'apim:llama-4-scout-17b', 'apim:deepseek-chat',
        'nero:gemini-2.0-flash', 'nero:gemini-2.5-pro', 'nero:gemini-2.5-flash',
        'nero:gemini-2.5-flash-lite'
    ]
    
    model_help = (
        "Default LLM model to use. Available models:\n"
        "  APIM models: apim:gpt-4.1, apim:gpt-4.1-mini, apim:gpt-4.1-nano, apim:o3-mini,\n"
        "               apim:claude-3.5, apim:claude-3.7, apim:gemini-2.0-flash,\n"
        "               apim:gemini-2.5-pro-preview-05-06, apim:llama-3.3-70b,\n"
        "               apim:llama-4-maverick-17b, apim:llama-4-scout-17b, apim:deepseek-chat\n"
        "  Nero models: nero:gemini-2.0-flash, nero:gemini-2.5-pro, nero:gemini-2.5-flash,\n"
        "               nero:gemini-2.5-flash-lite\n"
        "  Default: apim:gpt-4o-mini"
    )
    
    parser.add_argument(
        "--model",
        default="apim:gpt-4o-mini",
        help=model_help,
    )
    parser.add_argument(
        "--cache_dir",
        default=".cache",
        help="Directory to store the LLM response cache",
    )
    parser.add_argument(
        "--mcp_url", default="http://localhost:8000/mcp", help="MCP server URL"
    )
    # Use --patient_id 127672063
    parser.add_argument(
        "--patient_id", default=None, help="Patient ID to auto-load on startup"
    )
    return parser.parse_args()


# ========== Helper Functions ==========
def generate_system_prompt(query_date_str: str = None):
    """Generate system prompt with the specified date."""
    if query_date_str is None or query_date_str == "No data loaded":
        date_str = date.today().strftime("%B %d, %Y")
    else:
        try:
            # Try to parse and reformat the date
            from datetime import datetime
            parsed_date = datetime.strptime(query_date_str, "%Y-%m-%d %H:%M:%S")
            date_str = parsed_date.strftime("%B %d, %Y")
        except (ValueError, TypeError):
            # Fallback to provided string or current date
            date_str = query_date_str if query_date_str != "No data loaded" else date.today().strftime("%B %d, %Y")
    
    return f"""You are a helpful EHR assistant. **Today's date is {date_str}**.

You are given a patient's EHR data and a question. Your task is to answer the question **based solely on the provided EHR data**."""

# ========== Default Settings ==========
def get_defaults():
    """Get default configuration settings."""
    return {
        "model": "nero:gemini-2.0-flash",
        "system_prompt": generate_system_prompt(),
        "prompt_template":"""You are provided patient XML:

```
{context}
```

Using this information, answer:

```
{question}
```

Return a **single JSON object** exactly in this shape:

```json
{{
  "answer": "<short answer formatted in markdown, with inline uid citations>",
  "evidence": {{
    "<uid1>": ["<verbatim text snippet of supporting evidence for uid1>", ...],
    "<uid2>": ["<verbatim text snippet of supporting evidence for uid2>", ...],
    ...
  }}
}}
```

### Rules

1. **UID usage**
   Use the exact `uid` values from the XML `<event>` tags: `<patient_id>_event_<event_index>`.
   **Never** invent placeholders like `uid1`, `uid2`, etc.

2. **`answer` field**

   * Concise, markdown-formatted summary.
   * Include relevant dates/times.
   * Every factual statement must have an inline citation in the form `[[<uid>]]`.

3. **`evidence` field**

   * Keys: only the `<event>` `uid`s that *directly* support the answer.
   * Values: arrays of **verbatim substrings** from the XML (minimal span that proves the claim).

4. **Parsimony**
   Include only essential evidence. Avoid unrelated or redundant events.

---

**Example**

Instruction:
“Calculate 10-year cardiovascular risk using the most recent LDL, blood pressure, and smoking status.”

```json
{{
  "answer": "The 10-year cardiovascular risk cannot be calculated because the patient's LDL cholesterol level is not available in the provided records [[127672063_event_700]]. A lipid panel was ordered on July 17, 2022, but the results were not found during a chart review on that date [[127672063_event_700]].\n\nOther available risk factors include:\n* **Most Recent Blood Pressure**: 102/66 mmHg on November 19, 2022 [[127672063_event_727]].\n* **Smoking Status**: Former smoker, quit in 2018 [[127672063_event_700]].",
  "evidence": {{
    "127672063_event_700": [
      "No results found for: LDL, A1C, TSH",
      "Labs:  Lipid panel, A1c sent as external orders.",
      "Smoked 1/2 to 1 ppd x 27 years, quit 2018."
    ],
    "127672063_event_727": [
      "BP: 102/66 mmHg"
    ]
  }}
}}
```
""",
        "prompt_template_v1": """You are provided with relevant patient information below:

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
        "use_timeline": True,
        "use_cache": False,
        "use_streaming": True,
        "example_prompts": [
            "Calculate 10-year cardiovascular risk using the most recent LDL, blood pressure, and smoking status.",
            #"List all <event> tags of type 'image' with their corresponding imaging modality (e.g., CT, MRI, etc.), list in chronological order, include the date and anatomical location.",
            "List all medical imaging events (e.g., CT, MRI, etc.) in chronological order, include the modality (in bold), date of scan, and anatomical location.",
            "List inpatient admissions or ED visits in the past five years, with date, reason, and length of stay.",
            #"Extract mentions of housing instability, food insecurity, or transportation issues and their dates.",
            #"List current medications and any dose changes over the past six months.",
            "Outline a summary of this patient's cancer diagnoses, treatments, and corresponding responses. Include specific dates and evidence from the EHR.",
            TUMOR_BOARD_PROMPT
        ],
    }
