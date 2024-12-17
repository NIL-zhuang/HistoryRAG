from typing import List

from fastapi import Body, File, Form, UploadFile
from rag.server.specs import BaseResponse, Context, ContextMetadata, ListResponse
from rag.settings import Settings


def create_kb(
    kb_name: str = Body(..., example="default"),
    vector_store_type: str = Body(
        Settings.kb_settings.DEFAULT_VS_TYPE,
        description="Vector store type",
        examples=["faiss", "milvus"],
    ),
    kb_info: str = Body("", description="Description to knowledge base"),
    embed_model: str = Body("BAAI/bge-m3"),
) -> BaseResponse:
    pass


def drop_kb(kb_name: str = Body(example="default")) -> BaseResponse:
    pass


def list_kbs() -> ListResponse:
    pass


def upload_docs(
    files: List[UploadFile] = File(..., description="Uploaded files"),
    kb_name: str = Form(
        "default", description="Knowledge base name", example="default"
    ),
    override: bool = Form(False, description="Override existing documents"),
    chunk_size: int = Form(Settings.kb_settings.CHUNK_SIZE, description="Chunk size"),
    chunk_overlap: int = Form(Settings.kb_settings.OVERLAP_SIZE, description="Overlap"),
) -> BaseResponse:
    pass


def search(
    query: str = Body("", description="User query", example="Who is Zheyuan Lin"),
    knowledge_base_name: str = Body(
        "default", description="Knowledge base name", example="default"
    ),
    top_k: int = Body(
        Settings.kb_settings.VS_TOP_K, description="Top k retrieved chunks"
    ),
    score_threshold: float = Body(
        Settings.kb_settings.SCORE_THRESHOLD, description="Similarity score threshold"
    ),
) -> List[Context]:
    return [
        Context(
            meta_data=ContextMetadata(
                series_name="蒋介石日记",
                file_name="蒋介石日记.pdf",
                title="第一章",
                start_page=1,
                end_page=1,
            ),
            context="勃勃生机，万物竞发的境界，犹在眼前",
        )
    ]
