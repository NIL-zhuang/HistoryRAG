import json
from time import sleep
from typing import Callable, Dict, List, Union

import openai
from rag.server.llm.base import LLM
from rag.server.models.model_spec import ModelConfig
from rag.settings import Settings
from rag.utils import build_logger

SLEEP_SEC = 3
logger = build_logger()


class PlatformLLM(LLM):
    def __init__(self, model_config: ModelConfig):
        self.client = openai.Client(
            api_key=model_config.api_key,
            base_url=model_config.api_base_url,
        )
        self.model_config = model_config

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        return self._call(self._chat, messages=messages, **kwargs)

    def embed(self, content: str, **kwargs) -> List[float]:
        return self._call(self._embed, content=content, **kwargs)

    def _call(self, func: Callable, **kwargs) -> str:
        try:
            response = func(**kwargs)
            return response
        except openai.BadRequestError as e:
            err = json.loads(e.response.text)
            if err["error"]["code"] == "content_filter":
                logger.error("Content filter triggered!")
                return None
            logger.error(f"The OpenAI API request was invalid: {e}")
            return None
        except openai.APIConnectionError as e:
            logger.error(f"The OpenAI API connection failed: {e}")
            sleep(SLEEP_SEC)
            return func(**kwargs)
        except openai.RateLimitError as e:
            logger.error(
                f"Token rate limit exceeded. Retrying after {SLEEP_SEC} second..."
            )
            sleep(SLEEP_SEC)
            return func(**kwargs)
        except openai.APIError as e:
            if "The operation was timeout" in str(e):
                # Handle the timeout error here
                logger.error(
                    "The OpenAI API request timed out. Please try again later."
                )
                sleep(SLEEP_SEC)
                return func(**kwargs)
            elif "DeploymentNotFound" in str(e):
                logger.error("The API deployment for this resource does not exist")
                return None
            else:
                # Handle other API errors here
                logger.error(f"The OpenAI API returned an error: {e}")
                sleep(SLEEP_SEC)
                return func(**kwargs)
        except Exception as e:
            logger.error(f"An error occurred: {e}")

    def _chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = Settings.model_settings.TEMPERATURE,
        max_tokens: int = Settings.model_settings.MAX_TOKENS,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model_config.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def _embed(self, content: str) -> List[float]:
        embedding = self.client.embeddings.create(
            input=content,
            model=self.model_config.model_name,
        )
        return embedding.data[0].embedding
