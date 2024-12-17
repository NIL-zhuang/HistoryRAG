from typing import List, Union

from rag.server.models.kb_spec import Context
from rag.settings import Settings


class SupportedVectorStoreTypes:
    FAISS = "faiss"
    MILVUS = "milvus"


class KBService:
    def __init__(
        self,
        kb_name: str,
        kb_info: str = None,
        embed_model: str = Settings.model_settings.DEFAULT_EMBEDDING_MODEL,
    ):
        self.kb_name = kb_name
        self.kb_info = kb_info
        self.embed_model = embed_model

    def save_vector_store(self):
        """dump knowledge base to disk"""
        pass

    def create_kb(self, kb_name: str):
        """Create knowledge base"""
        pass

    def drop_kb(self):
        """Delete knowledge base"""
        pass

    def add_doc(self):
        pass

    def delete_doc(self):
        pass

    def update_doc(self):
        pass

    def search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.3,
    ) -> List[Context]:
        pass


class KBServiceFactory:
    @staticmethod
    def get_kb_service(
        kb_name: str,
        vector_store_type: Union[str, SupportedVectorStoreTypes],
        embed_model: str = Settings.model_settings.DEFAULT_EMBEDDING_MODEL,
        kb_info: str = None,
    ) -> KBService:
        if isinstance(vector_store_type, str):
            vector_store_type = getattr(
                SupportedVectorStoreTypes, vector_store_type.upper()
            )
        if vector_store_type == SupportedVectorStoreTypes.MILVUS:
            from rag.server.kb.milvus_kb_service import MilvusKBService

            return MilvusKBService(kb_name, kb_info, embed_model)
