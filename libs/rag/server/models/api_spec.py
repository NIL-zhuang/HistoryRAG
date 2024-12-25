from typing import Any, List, Optional

from rag.server.pydantic_v2 import BaseModel, Field


class KBRequest(BaseModel):
    kb_name: str = Field(description="Knowledge Base name")


class BaseResponse(BaseModel):
    code: int = Field(200, description="API status code")
    msg: str = Field("success", description="API status message")
    data: Optional[Any] = Field(None, description="API data")

    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }


class ListResponse(BaseResponse):
    data: Optional[List[Any]] = Field(None, description="List of data")

    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
                "data": ["doc1.docx", "doc2.pdf", "doc3.txt"],
            }
        }
