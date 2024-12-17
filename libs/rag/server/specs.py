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


class ContextMetadata(BaseModel):
    series_name: str = Field(description="Series name")
    file_name: str = Field(description="File name")
    title: str = Field(description="Title")
    start_page: int = Field(description="Content start page")
    end_page: int = Field(description="Content end page")


class Context(BaseModel):
    meta_data: ContextMetadata = Field(description="Context metadata")
    context: str = Field(description="Context content")


class BaseResponse(BaseModel):
    code: int = Field(200, description="API status code")
    msg: str = Field("success", description="API status message")
    data: Any = Field(None, description="API data")

    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }


class ListResponse(BaseResponse):
    data: List[Any] = Field(..., description="List of data")

    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
                "data": ["doc1.docx", "doc2.pdf", "doc3.txt"],
            }
        }


class ModelConfig(BaseModel):
    platform_name: str = Field("silicon-flow", description="Platform name")
    platform_type: str = Field("openai", description="Platform type")
    api_base_url: str = Field(
        "https://api.siliconflow.cn/v1", description="API base URL"
    )
    api_key: str = Field("sk-xxx", description="API key")
    model_name: str = Field("deepseek-ai/DeepSeek-V2.5", description="Model name")
