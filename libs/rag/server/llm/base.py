from typing import Dict, List, Union

from rag.server.llm.utils import get_model_configs


class LLM:
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        raise NotImplementedError

    def embed(self, contents: Union[List[str], str], **kwargs) -> List[List[float]]:
        raise NotImplementedError


class LLMFactory:
    @staticmethod
    def get_llm_service(model_name: str, platform_name: str = None) -> LLM:
        model_config = get_model_configs(model_name, platform_name)
        from rag.server.llm.proxy_llm import PlatformLLM

        llm_service = PlatformLLM(model_config)
        return llm_service
