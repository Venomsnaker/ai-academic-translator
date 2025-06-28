from openai import OpenAI
import time

class OpenAIClient:
    def __init__(
            self,
            api_key,
            model = 'gpt-4.1-nano-2025-04-14',
            retries = 3):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.retries = retries

    def generate_response(self, prompt: str, temeprature = 1):
        retries = 3

        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': prompt}
                    ],
                    temperature=temeprature,
                )
                return [choice.message.content for choice in response.choices]
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)
                else:
                    raise e