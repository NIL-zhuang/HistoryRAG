from fastapi import APIRouter
from rag.server.kb.kb_api import *
from rag.server.models.api_spec import BaseResponse, ListResponse

kb_router = APIRouter(prefix="/kb", tags=["Knowledge Base"])

kb_router.post("/create_kb", response_model=BaseResponse)(create_kb)
kb_router.post("/drop_kb", response_model=BaseResponse)(drop_kb)
kb_router.get("/list_kbs", response_model=ListResponse)(list_kbs)

kb_router.get("/list_collection", response_model=ListResponse)(list_collection)
kb_router.post("/create_collection", response_model=BaseResponse)(create_collection)
kb_router.post("/drop_collection", response_model=BaseResponse)(drop_collection)

kb_router.post("/upload_docs", response_model=BaseResponse)(upload_docs)
kb_router.post("/add_context", response_model=BaseResponse)(add_context)

kb_router.post("/search", response_model=ListResponse)(search)
