from typing import List, Dict, Any

from fastapi import Body

from rag.server.kb.kb_api import search, full_text_search
from rag.server.models.kb_spec import Context
from rag.server.retrieve import reranker


def get_docs(
        rewrite_res: Dict[str, Any],
        kb_name: str = Body(None, description="Knowledge base name", example="default"),
        collection_name: str = Body(None, description="Collection name"),
        collection_name_for_sparse: str = Body(None, description="Collection name for sparse"),
        top_k: int = Body(5, description="Nums of matched vectors"),
        score_threshold: float = Body(
        0.1, description="Threshold of matched vectors", ge=0, le=1
    ),
) -> List[Context]:
    raw_query = rewrite_res['raw_query']
    optimized_queries = rewrite_res['optimized_queries']
    keywords = rewrite_res['keywords']

    docs = []
    docs.append(search(raw_query, kb_name, collection_name, top_k, score_threshold).data)
    for query in optimized_queries:
        docs.append(search(query, kb_name, collection_name, top_k, score_threshold).data)

    if collection_name_for_sparse is not None and len(keywords) > 0:
        full_text_query = ""
        for kw in keywords:
            full_text_query += " "
            full_text_query += kw
        keywords = [full_text_query]
        docs.append(full_text_search(keywords, kb_name, collection_name_for_sparse, top_k, score_threshold).data)

    docs = reranker.rrf(docs)
    return docs[:top_k]