from typing import Any, Dict, List, Tuple, Union

from rag.server.pydantic_v2 import BaseModel, Field


class History(BaseModel):
    role: str = Field()
    content: str = Field()

    @classmethod
    def from_data(cls, history: Union[List, Tuple, Dict]) -> "History":
        if isinstance(history, (list, tuple)) and len(history) >= 2:
            return cls(role=history[0], content=history[1])
        elif isinstance(history, dict):
            return cls(**history)


class ModelConfig(BaseModel):
    platform_name: str = Field("silicon-flow", description="Platform name")
    platform_type: str = Field("openai", description="Platform type")
    api_base_url: str = Field(
        "https://api.siliconflow.cn/v1", description="API base URL"
    )
    api_key: str = Field("sk-xxx", description="API key")
    model_name: str = Field("deepseek-ai/DeepSeek-V2.5", description="Model name")
    meta_data: Dict[str, Any] = Field({}, description="Meta data")
