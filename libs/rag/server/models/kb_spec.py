from typing import Optional, Union

from rag.server.pydantic_v2 import BaseModel, Field


class ContextMetadata(BaseModel):
    series_name: Optional[str] = Field(default=None, description="Series name")
    file_name: Optional[str] = Field(default=None, description="File name")
    title: Optional[str] = Field(default=None, description="Title")
    start_page: Optional[int] = Field(default=None, description="Content start page")
    end_page: Optional[int] = Field(default=None, description="Content end page")


class Context(BaseModel):
    id: Optional[Union[str, int]] = Field(default=None, description="Context ID")
    distance: Optional[float] = Field(default=None, description="Similarity Distance")
    metadata: ContextMetadata = Field(description="Context metadata")
    content: str = Field(description="Context content")
