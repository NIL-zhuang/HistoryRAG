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
    COLLECTION_NAME: str = "default"
    DEFAULT_VS_TYPE: Literal["faiss", "milvus"] = "milvus"
    CHUNK_SIZE: int = 1024
    OVERLAP_SIZE: int = 200
    VS_TOP_K: int = 10
    SCORE_THRESHOLD: float = 0.3


class PromptSettings(BaseFileSettings):
    model_config = SettingsConfigDict(yaml_file=CONFIG_ROOT / "prompt_configs.yaml")
    DEFAULT_SYSTEM_PROMPT: str = "You are a helpful assistant."

    RAG_PROMPT: dict = {
        "default": [
            {"role": "system", "content": "你是一个文档问答小助手"},
            {
                "role": "user",
                "content": (
                    "根据提供的参考资料，回答问题。如果找不到答案，请不要编造内容，可以回答“不知道”。\n"
                    "<<已知信息>>\n"
                    "{{context}}\n"
                    "<<问题>>\n"
                    "{{query}}",
                ),
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
