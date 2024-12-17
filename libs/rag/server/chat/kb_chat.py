from typing import List

from fastapi import Body
from rag.server.chat.llm.base import LLMFactory
from rag.server.chat.utils import construct_message
from rag.server.kb.kb_api import search
from rag.server.specs import History
from rag.settings import Settings
from rag.utils import build_logger

logger = build_logger()


async def kb_chat(
    query: str = Body(
        description="Query to chat with knowledge base",
        example="Who is zheyuan lin?",
    ),
    kb_name: str = Body(
        "default", description="Knowledge base name", example="default"
    ),
    top_k: int = Body(5, description="Nums of matched vectors"),
    score_threshold: float = Body(
        0.1, description="Threshold of matched vectors", ge=0, le=1
    ),
    history: List[History] = Body(
        [],
        description="History of chat",
        example=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hello, what can I do for you?"},
        ],
    ),
    stream: bool = Body(False, description="Stream response"),
    model: str = Body(
        Settings.model_settings.DEFAULT_LLM_NAME, description="Model name"
    ),
    temperature: float = Body(0.6, description="Temperature of sampling"),
    max_tokens: int = Body(2000, description="Max tokens of sampling"),
    prompt_name: str = Body("default", description="Prompt template name"),
):
    """
    Knowledge base chat
    """
    logger.info(f"User query: {query}")
    history = [History.from_data(h) for h in history]
    docs = search(query, kb_name, top_k, score_threshold)
    prompt_template = Settings.prompt_settings.RAG_PROMPT[prompt_name]
    llm = LLMFactory.get_llm_service(model)
    messages = construct_message(query, history, docs, prompt_template)
    response = llm.chat(messages, temperature=temperature, max_tokens=max_tokens)
    logger.info(f"Model response: {response}")
    return response
