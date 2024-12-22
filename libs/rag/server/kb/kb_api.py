from typing import List

from fastapi import Body, File, Form, UploadFile
from rag.server.kb.base import KBServiceFactory
from rag.server.models.api_spec import BaseResponse, ListResponse
from rag.server.models.kb_spec import Context, ContextMetadata
from rag.settings import Settings
from rag.utils import build_logger

logger = build_logger()


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
    if kb_name is None or kb_name.strip() == "":
        return BaseResponse(status="403", message="Knowledge base name is empty")

    kb = KBServiceFactory.get_kb_service_by_name(kb_name)
    if kb is not None:
        return BaseResponse(code=403, msg="Knowledge base already exists")

    kb = KBServiceFactory.get_kb_service(
        kb_name, vector_store_type, embed_model, kb_info
    )
    try:
        kb.create_kb(kb_name)
    except Exception as e:
        msg = f"Fail to create knowledge base: {e}"
        logger.error(f"{e.__class__.__name__}: {msg}")
        return BaseResponse(code=500, msg=msg)

    return BaseResponse(code=200, msg="Knowledge base created")

def drop_kb(kb_name: str = Body(example="default")) -> BaseResponse:
    kb = KBServiceFactory.get_kb_service_by_name(kb_name)
    if kb is None:
        return BaseResponse(code=404, msg="Knowledge base not found")

    try:
        kb.drop_kb()
    except Exception as e:
        msg = f"Fail to drop knowledge base: {e}"
        logger.error(f"{e.__class__.__name__}: {msg}")
        return BaseResponse(code=500, msg=msg)


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
    raise NotImplementedError


def upload_context(
    context: Context = Body(..., description="Context to upload"),
    kb_name: str = Body(
        "default", description="Knowledge base name", example="default"
    ),
) -> BaseResponse:
    kb = KBServiceFactory.get_kb_service_by_name(kb_name)
    if kb is None:
        return BaseResponse(code=404, msg="Knowledge base not found")
    try:
        kb.add_context(context)
    except Exception as e:
        msg = f"Fail to upload context: {e}"
        logger.error(f"{e.__class__.__name__}: {msg}")
        return BaseResponse(code=500, msg=msg)
    return BaseResponse(code=200, msg="Context uploaded")


def search(
    query: str = Body("", description="User query", example="Who is Zheyuan Lin"),
    knowledge_base_name: str = Body(
        "default", description="Knowledge base name", example="default"
    ),
    collection_name: str = Body(
        "default", description="Collection name", example="default"
    ),
    top_k: int = Body(
        Settings.kb_settings.VS_TOP_K, description="Top k retrieved chunks"
    ),
    score_threshold: float = Body(
        Settings.kb_settings.SCORE_THRESHOLD, description="Similarity score threshold"
    ),
) -> List[Context]:
    kb = KBServiceFactory.get_kb_service_by_name(knowledge_base_name)
    if kb is None:
        return []
    if query is None or query.strip() == "":
        return []
    contexts = kb.search(query, collection_name, top_k, score_threshold)
    return contexts
