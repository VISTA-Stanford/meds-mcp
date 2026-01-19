from .vignette_base import BaseVignetteGenerator


class LLMVignetteGenerator(BaseVignetteGenerator):
    def __init__(self, base_generator: BaseVignetteGenerator, llm):
        """
        llm must expose:
          summarize(text: str) -> str
        """
        self.base = base_generator
        self.llm = llm

    def generate(self, *args, **kwargs) -> str:
        base_text = self.base.generate(*args, **kwargs)
        if not base_text.strip():
            return base_text
        return self.llm.summarize(base_text)
