import json
from time import sleep
from typing import Dict, List

import openai
from rag.server.chat.llm.base import LLM
from rag.server.specs import ModelConfig

SLEEP_SEC = 3


class ProxyLLM(LLM):
    def __init__(self, model_config: ModelConfig):
        self.client = openai.Client(
            api_key=model_config.api_key,
            base_url=model_config.api_base_url,
        )
        self.model_config = model_config

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            response = self._chat(messages, **kwargs)
            return response
        except openai.BadRequestError as e:
            err = json.loads(e.response.text)
            if err["error"]["code"] == "content_filter":
                print("Content filter triggered!")
                return None
            print(f"The OpenAI API request was invalid: {e}")
            return None
        except openai.APIConnectionError as e:
            print(f"The OpenAI API connection failed: {e}")
            sleep(SLEEP_SEC)
            return self.chat(messages, **kwargs)
        except openai.RateLimitError as e:
            print(f"Token rate limit exceeded. Retrying after {SLEEP_SEC} second...")
            sleep(SLEEP_SEC)
            return self.chat(messages, **kwargs)
        except openai.AuthenticationError as e:
            print(f"Invalid API token: {e}")
            self.update_api_key()
            sleep(SLEEP_SEC)
            return self.chat(messages, **kwargs)
        except openai.APIError as e:
            if "The operation was timeout" in str(e):
                # Handle the timeout error here
                print("The OpenAI API request timed out. Please try again later.")
                sleep(SLEEP_SEC)
                return self.chat(messages, **kwargs)
            elif "DeploymentNotFound" in str(e):
                print("The API deployment for this resource does not exist")
                return None
            else:
                # Handle other API errors here
                print(f"The OpenAI API returned an error: {e}")
                sleep(SLEEP_SEC)
                return self.chat(messages, **kwargs)
        except Exception as e:
            print(f"An error occurred: {e}")

    def _chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model_config.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
