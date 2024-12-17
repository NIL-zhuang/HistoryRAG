from typing import Dict, List

from rag.server.utils import get_model_configs


class LLM:
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        raise NotImplementedError


class LLMFactory:
    @staticmethod
    def get_llm_service(model_name: str, platform_name: str = None) -> LLM:
        model_config = get_model_configs(model_name, platform_name)
        from rag.server.chat.llm.proxy_llm import ProxyLLM

        llm_service = ProxyLLM(model_config)
        return llm_service
