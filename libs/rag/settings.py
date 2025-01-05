from __future__ import annotations

import os
from pathlib import Path
from typing import *

from pydantic_settings import SettingsConfigDict
from rag import __version__
from rag.pydantic_settings_file import *

RAG_ROOT = Path(os.environ.get("RAG_ROOT", ".")).resolve()
CONFIG_ROOT = RAG_ROOT / "configs"


class BasicSettings(BaseFileSettings):
    model_config = SettingsConfigDict(yaml_file=CONFIG_ROOT / "basic_configs.yaml")
    log_verbose: bool = False
    KB_ROOT_PATH: str = str(RAG_ROOT / "data/knowledge_base")
    version: str = __version__

    @cached_property
    def DATA_PATH(self):
        return RAG_ROOT / "data"

    @cached_property
    def LOG_PATH(self):
        return self.DATA_PATH / "logs"

    @cached_property
    def TMP_PATH(self):
        return self.DATA_PATH / "tmp"

    def make_dirs(self):
        """创建所有数据目录"""
        for p in [self.DATA_PATH, self.LOG_PATH, self.KB_ROOT_PATH]:
            p.mkdir(parents=True, exist_ok=True)


class PlatformConfig(MyBaseModel):
    PLATFORM_NAME: str = "silicon-flow"
    PLATFORM_TYPE: Literal["openai"] = "openai"
    API_BASE_URL: str = "https://api.siliconflow.cn/v1"
    API_KEY: str = "sk-xxx"
    LLM_MODELS: Dict[str, Dict[str, Any]] = []
    EMBEDDING_MODELS: Dict[str, Dict[str, Any]] = []
    RERANK_MODELS: Dict[str, Dict[str, Any]] = []


class ModelSettings(BaseFileSettings):
    model_config = SettingsConfigDict(yaml_file=CONFIG_ROOT / "model_configs.yaml")
    DEFAULT_LLM_NAME: str = "deepseek-ai/DeepSeek-V2.5"
    DEFAULT_EMBEDDING_MODEL: str = "BAAI/bge-m3"
    DEFAULT_RERANK_MODEL: str = "BAAI/bge-reranker-v3-m3"
    HISTORY_LEN: int = 5
    MAX_TOKENS: Optional[int] = 2e5
    TEMPERATURE: float = 0.3
    MODEL_PLATFORMS: List[PlatformConfig] = [
        PlatformConfig(
            **{
                "PLATFORM_NAME": "silicon-flow",
                "PLATFORM_TYPE": "openai",
                "API_BASE_URL": "https://api.siliconflow.cn/v1",
                "API_KEY": "sk-xxx",
                "LLM_MODELS": {
                    "deepseek-ai/DeepSeek-V2.5": {},
                    "Qwen/Qwen-2.5-72B-Instruct": {},
                },
                "EMBEDDING_MODELS": {"BAAI/bge-m3": {"embed_size": 1024}},
                "RERANK_MODELS": {"BAAI/bge-reranker-v3-m3": {}},
            }
        )
    ]


class KBSettings(BaseFileSettings):
    model_config = SettingsConfigDict(yaml_file=CONFIG_ROOT / "kb_configs.yaml")
    MILVUS_HOST: str = "http://localhost:19530"
    MILVUS_TOKEN: str = "root:Milvus"
    DEFAULT_COLLECTION_NAME: str = "default"
    DEFAULT_VS_TYPE: Literal["faiss", "milvus"] = "milvus"
    CHUNK_SIZE: int = 1024
    OVERLAP_SIZE: int = 200
    VS_TOP_K: int = 10
    SCORE_THRESHOLD: float = 0.0


# 关于prompt_name
#
# pre-generation: 不提供文献资料，直接要求模型给出问题的回答。可作为对照组用于测试和评估
# weak-reference: 模型的回答与文献内容存在较弱的关联性，模型给出的回复会在一定程度上参考文献资料。当给定的文档与问题完全无关时，
#                 会退化为 pre-generation 模式
# default: 默认采用此方案，介于 weak-reference 和 strong-reference 之间，模型会充分参照文献内容给出相应的回答
# strong-reference: 模型将完全按照文献中的内容回答，不产生额外的见解
class PromptSettings(BaseFileSettings):
    model_config = SettingsConfigDict(yaml_file=CONFIG_ROOT / "prompt_configs.yaml")
    DEFAULT_SYSTEM_PROMPT: str = "You are a helpful assistant."

    RAG_PROMPT : dict = {
        "pre-generation": [
            {
                "role": "system",
                "content": (
                    "你是一个知识问答小助手。\n"
                    "根据你对问题的理解，给出你的回答。\n"
                    "<<这些是历史问答记录>>"
                    "{history}"
                )},
            {
                "role": "user",
                "content": (
                    "{query}"
                )
            },
        ],
        "weak-reference":  [
            {
                "role": "system",
                "content": (
                    "你是一个文档问答小助手，会参照文档资料中的内容给出问题的答复。\n"
                    "用户现在提出了一些问题，你的任务是给出你的答复。\n"
                    "为了帮助你更好的回答，我们还为你准备了一些可能有关的资料：\n"
                    "<<已知信息>>\n"
                    "{contexts}\n"
                    "当你准备给出答复时，如果已知资料中的内容与你的见解存在相似之处，则你可以充分利用这些内容来润色你的回复。当然，这些资料"
                    "中的内容未必准确，因此，如果其中存在与你的见解相悖或无关的内容，还应以你的判断为主。"
                    "<<这些是历史问答记录>>"
                    "{history}"
                )},
            {
                "role": "user",
                "content": (
                    "{query}\n",
                )
            },
        ],
        "default": [
            {
                "role": "system",
                "content": (
                    "你是一个文档问答小助手，会参照文档资料中的内容给出问题的答复。\n"
                    "你应该根据提供的参考资料，回答问题。\n"
                    "<<已知信息>>\n"
                    "{contexts}\n"
                    "你可以根据这些资料的内容来回答问题，可以同时存在一些自己的见解或补充。如果已知资料中的内容与你的见解存在相似之处，"
                    "则你可以将这部分内容与你的观点充分融合；反之如果差异较大，则你可以自行判断孰优孰劣。特别地，如果完全无法从已知资料"
                    "中找到相关的信息，则不太建议你随意编造内容，你可以回答自己并不知晓。\n"
                    "<<这些是历史问答记录>>"
                    "{history}"
                )},
            {
                "role": "user",
                "content": (
                    "{query}"
                )
            },
        ],
        "strong-reference": [
            {
                "role": "system",
                "content": (
                    "你是一个文档问答小助手，会归纳总结文档资料中的内容，并根据这些内容给出问题的答复。\n"
                    "这里是你所掌握的全部已知信息：\n"
                    "<<已知信息>>\n"
                    "{contexts}\n"
                    "根据以上资料，回答用户提出的问题。注意：请严格按照资料中给出的信息进行回答，你可以对其中相关的内容进行总结、概括、转述，"
                    "但你回复的所有内容都应当能够从资料中找到相似的内容或出处。如果不能在资料中找到合适的答案，你应该直接表示自己并不知晓。\n"
                    "<<这些是历史问答记录>>"
                    "{history}"
                )},
            {
                "role": "user",
                "content": (
                    "{query}"
                )
            },
        ]
    }

class SettingsContainer:
    ROOT_PATH = RAG_ROOT
    basic_settings: BasicSettings = settings_property(BasicSettings())
    model_settings: ModelSettings = settings_property(ModelSettings())
    kb_settings: KBSettings = settings_property(KBSettings())
    prompt_settings: PromptSettings = settings_property(PromptSettings())

    def create_templates(self):
        self.basic_settings.create_template_file(write_file=True)
        self.model_settings.create_template_file(write_file=True)
        self.kb_settings.create_template_file(write_file=True)
        self.prompt_settings.create_template_file(write_file=True)


Settings = SettingsContainer()

if __name__ == "__main__":
    Settings.create_templates()
