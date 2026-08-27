from deepeval.models import DeepEvalBaseLLM
import anthropic

class ClaudeEvalModel(DeepEvalBaseLLM):
    def __init__(self):
        self.client = anthropic.Anthropic()

    def load_model(self): return self

    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self): return "claude-sonnet-4-6"