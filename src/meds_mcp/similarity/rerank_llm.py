class LLMSimilarityReranker:
    """
    LLM-based reranker using pairwise similarity judgment.
    """

    def __init__(self, llm):
        self.llm = llm

    def score(self, query_vignette: str, candidate_vignette: str) -> float:
        prompt = f"""
You are comparing two patient clinical vignettes.

Task:
Score how clinically similar they are on a scale from 0 to 10.

Rules:
- Consider diagnoses, treatments, procedures, and disease stage.
- Ignore writing style.
- Do not add new facts.
- Return ONLY a number.

Vignette A:
{query_vignette}

Vignette B:
{candidate_vignette}
"""
        response = self.llm.summarize(prompt)

        try:
            return float(response.strip())
        except Exception:
            return 0.0
