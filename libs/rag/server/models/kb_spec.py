from rag.server.pydantic_v2 import BaseModel, Field


class ContextMetadata(BaseModel):
    series_name: str = Field(description="Series name")
    file_name: str = Field(description="File name")
    title: str = Field(description="Title")
    start_page: int = Field(description="Content start page")
    end_page: int = Field(description="Content end page")


class Context(BaseModel):
    meta_data: ContextMetadata = Field(description="Context metadata")
    context: str = Field(description="Context content")
