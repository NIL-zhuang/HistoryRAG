from typing import List

from fastapi import APIRouter, Request
from rag.server.kb.kb_api import create_kb, drop_kb, list_kbs, search, upload_docs
from rag.server.specs import BaseResponse, ListResponse

kb_router = APIRouter(prefix="/kb", tags=["Knowledge Base"])

kb_router.get(
    "/list_kbs", response_model=ListResponse, summary="Get all knowledge bases."
)(list_kbs)

kb_router.post(
    "/create_kb", response_model=BaseResponse, summary="Create knowledge base"
)(create_kb)

kb_router.post("/drop_kb", response_model=BaseResponse, summary="Drop knowledge base")(
    drop_kb
)

kb_router.post("/search", response_model=List[dict], summary="Search knowledge base")(
    search
)
