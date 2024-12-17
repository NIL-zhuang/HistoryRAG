from fastapi import APIRouter, Request
from rag.server.chat.kb_chat import kb_chat

chat_router = APIRouter(prefix="/chat", tags=["chat"])
chat_router.post("/kb_chat", summary="knowledge base chat")(kb_chat)
