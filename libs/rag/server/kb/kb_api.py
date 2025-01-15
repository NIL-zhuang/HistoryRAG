from typing import List, Union

from fastapi import Body, File, Form, UploadFile
from rag.server.api_server.utils import (
    filter_collection_with_kb_name,
    map_collection_name,
)
from rag.server.kb.base import KBServiceFactory
from rag.server.models.api_spec import BaseResponse, KBRequest, ListResponse
from rag.server.models.kb_spec import Context
from rag.settings import Settings
from rag.utils import build_logger
from rag.server.db.kb_repo_impl import MongoDB

logger = build_logger()

__all__ = [
    "create_kb",
    "drop_kb",
    "list_kbs",
    "create_collection",
    "drop_collection",
    "list_collection",
    "upload_docs",
    "add_context",
    "search",
]


def create_kb(
    kb_name: str = Body(example="default"),
    kb_info: str = Body("", description="Description to knowledge base"),
    vector_store_type: str = Body(
        Settings.kb_settings.DEFAULT_VS_TYPE,
        description="Vector store type",
        examples=["faiss", "milvus"],
    ),
    embed_model: str = Body("BAAI/bge-m3"),
) -> BaseResponse:
    if kb_name is None or kb_name.strip() == "":
        return BaseResponse(status="403", message="Knowledge base name is empty")

    # kb = KBServiceFactory.get_kb_service_by_name(kb_name)
    kb = None  # TODO: fix this when DB is implemented
    if kb is not None:
        return BaseResponse(code=403, msg="Knowledge base already exists")

    kb = KBServiceFactory.get_kb_service(
        kb_name, kb_info, vector_store_type, embed_model
    )
    try:
        # TODO: add collection info to db
        MongoDB.connect_to_db()
        # 需要对list_collection添加 kb_name 之后来记录
        MongoDB.insert_document("", "", kb.list_collection())
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
        collections = kb.list_collection()
        kb_collections = filter_collection_with_kb_name(kb_name, collections)
        for c in kb_collections:
            kb.drop_collection(c)
        # TODO: remove collections from db
        MongoDB.connect_to_db()
        for collection in kb_collections:
            MongoDB.delete_documents("", "", {"collection_name": collection})
    except Exception as e:
        msg = f"Fail to drop knowledge base: {e}"
        logger.error(f"{e.__class__.__name__}: {msg}")
        return BaseResponse(code=500, msg=msg)
    return BaseResponse(code=200, msg="Knowledge base dropped")

def list_kbs() -> ListResponse:
    # TODO: search DB for all knowledge bases
    pass


def create_collection(
    kb_name: str = Body(description="Knowledge base name", example="default"),
    collection_name: str = Body(description="Collection name", example="history_rag"),
    collection_info: str = Body("", description="Description to collection"),
) -> BaseResponse:
    kb = KBServiceFactory.get_kb_service_by_name(kb_name)
    if kb is None:
        return BaseResponse(code=404, msg="Knowledge base not found")
    try:
        collection_name = map_collection_name(kb_name, collection_name)
        if collection_name in kb.list_collection():
            return BaseResponse(code=403, msg="Collection already exists")
        kb.create_collection(collection_name, collection_info)
        # TODO: add collection to db
    except Exception as e:
        msg = f"Fail to create collection: {e}"
        logger.error(f"{e.__class__.__name__}: {msg}")
        return BaseResponse(code=500, msg=msg)
    return BaseResponse(code=200, msg="Collection created")


def drop_collection(
    kb_name: str = Body(description="Knowledge base name", example="default"),
    collection_name: str = Body(description="Collection name", example="history_rag"),
) -> BaseResponse:
    # TODO: remove collection from db
    kb = KBServiceFactory.get_kb_service_by_name(kb_name)
    if kb is None:
        return BaseResponse(code=404, msg="Knowledge base not found")
    try:
        collection_name = map_collection_name(kb_name, collection_name)
        if collection_name not in kb.list_collection():
            return BaseResponse(code=404, msg="Collection not exist")
        kb.drop_collection(collection_name)
    except Exception as e:
        msg = f"Fail to drop collection: {e}"
        logger.error(f"{e.__class__.__name__}: {msg}")
        return BaseResponse(code=500, msg=msg)
    return BaseResponse(code=200, msg="Collection dropped")


def list_collection(
    body: KBRequest = Body(description="Knowledge base name", example="default"),
) -> ListResponse:
    kb_name = body.kb_name
    kb = KBServiceFactory.get_kb_service_by_name(kb_name)
    if kb is None:
        msg = f"Knowledge base {kb_name} not found"
        logger.error(msg)
        return ListResponse(code=404, msg=msg)
    try:
        collections = kb.list_collection()
        collections = filter_collection_with_kb_name(kb_name, collections)
    except Exception as e:
        msg = f"Fail to list collections: {e}"
        logger.error(f"{e.__class__.__name__}: {msg}")
        return ListResponse(code=500, msg=msg)
    return ListResponse(code=200, msg="Collections listed", data=collections)


def upload_docs(
    files: List[UploadFile] = File(description="Uploaded files"),
    kb_name: str = Form(
        "default", description="Knowledge base name", example="default"
    ),
    override: bool = Form(False, description="Override existing documents"),
    chunk_size: int = Form(Settings.kb_settings.CHUNK_SIZE, description="Chunk size"),
    chunk_overlap: int = Form(Settings.kb_settings.OVERLAP_SIZE, description="Overlap"),
) -> BaseResponse:
    raise NotImplementedError


def add_context(
    context: Union[Context, List[Context]] = Body(..., description="Context to upload"),
    kb_name: str = Body(
        "default", description="Knowledge base name", example="default"
    ),
    collection_name: str = Body(
        "history_rag", description="Collection name", example="history_rag"
    ),
) -> BaseResponse:
    kb = KBServiceFactory.get_kb_service_by_name(kb_name)
    if kb is None:
        return BaseResponse(code=404, msg="Knowledge base not found")
    try:
        collection_name = map_collection_name(kb_name, collection_name)
        kb.add_context(collection_name, context)
    except Exception as e:
        msg = f"Fail to upload context: {e}"
        logger.error(f"{e.__class__.__name__}: {msg}")
        return BaseResponse(code=500, msg=msg)
    return BaseResponse(code=200, msg="Context uploaded")


def search(
    query: str = Body("", description="User query", example="Who is Zheyuan Lin"),
    kb_name: str = Body(
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
) -> ListResponse:
    kb = KBServiceFactory.get_kb_service_by_name(kb_name)
    if kb is None:
        return ListResponse(code=404, msg="Knowledge base not found")
    if query is None or query.strip() == "":
        return ListResponse(code=400, msg="Query is empty")
    try:
        collection_name = map_collection_name(kb_name, collection_name)
        contexts = kb.search(query, collection_name, top_k, score_threshold)
    except Exception as e:
        msg = f"Fail to search query {query}: {e}"
        logger.error(f"{e.__class__.__name__}: {msg}")
        return ListResponse(code=500, msg=msg)
    return ListResponse(code=200, msg="Search results", data=contexts)
