from rag.server.pydantic_v2 import BaseModel, Field


class ContextMetadata(BaseModel):
    series_name: str = Field(default=None, description="Series name")
    file_name: str = Field(default=None, description="File name")
    title: str = Field(default=None, description="Title")
    start_page: int = Field(default=None, description="Content start page")
    end_page: int = Field(default=None, description="Content end page")


class Context(BaseModel):
    meta_data: ContextMetadata = Field(description="Context metadata")
    content: str = Field(description="Context content")
