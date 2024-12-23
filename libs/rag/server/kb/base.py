from abc import ABC, abstractmethod
from typing import List, Union

from rag.server.models.kb_spec import Context
from rag.settings import Settings


class SupportedVectorStoreTypes:
    FAISS = "faiss"
    MILVUS = "milvus"


class KBService(ABC):
    def __init__(
        self,
        kb_name: str,
        kb_info: str = None,
        embed_model: str = Settings.model_settings.DEFAULT_EMBEDDING_MODEL,
    ):
        self.kb_name = kb_name
        self.kb_info = kb_info
        self.embed_model = embed_model

    def create_kb(
        self,
        kb_name: str,
        kb_info: str,
        embed_model: str = Settings.model_settings.DEFAULT_EMBEDDING_MODEL,
    ):
        # TODO: insert kb info to db
        self.do_create_kb(kb_name, kb_info, embed_model)

    def drop_kb(self, kb_name: str):
        # TODO: delete kb info from db
        pass

    def add_doc(self):
        # TODO 1. parse document into Context list contexts
        contexts = []
        for context in contexts:
            self.add_context(context)

    def delete_doc(self):
        # TODO search context by doc metadata
        # TODO delete context from vector store
        pass

    def update_doc(self):
        # TODO search context by doc metadata
        # TODO update context in vector store
        pass

    @abstractmethod
    def create_collection(
        self,
        collection_name: str,
        collection_info: str = "",
        **kwargs,
    ):
        """Create KB collection"""

    @abstractmethod
    def drop_collection(self, collection_name: str):
        """Delete knowledge base"""

    @abstractmethod
    def do_create_kb(self, kb_name: str, kb_info: str, embed_model: str):
        """Create knowledge base"""

    @abstractmethod
    def list_collection(self):
        """List all collections in the knowledge base"""

    @abstractmethod
    def add_context(self, collection_name: str, context: Context):
        """Add context to kb collection"""

    @abstractmethod
    def search(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
        score_threshold: float = 0.3,
        **kwargs,
    ) -> List[Context]:
        """Search for similar contexts in kb"""

    @abstractmethod
    def create_collection(self, collection_name: str, collection_info: str, **kwargs):
        pass

    @abstractmethod
    def save_vector_store(self):
        """dump knowledge base to disk"""


class KBServiceFactory:
    @staticmethod
    def get_kb_service(
        kb_name: str,
        kb_info: str,
        vector_store_type: Union[str, SupportedVectorStoreTypes],
        embed_model: str = Settings.model_settings.DEFAULT_EMBEDDING_MODEL,
    ) -> KBService:
        if isinstance(vector_store_type, str):
            vector_store_type = getattr(
                SupportedVectorStoreTypes, vector_store_type.upper()
            )
        if vector_store_type == SupportedVectorStoreTypes.MILVUS:
            from rag.server.kb.milvus_kb_service import MilvusKBService

            return MilvusKBService(kb_name, kb_info, embed_model)
        return None

    @staticmethod
    def get_kb_service_by_name(kb_name: str) -> KBService:
        # TODO: get kb info from db
        # dummy code, will fail on create_kb
        vector_store_type = "milvus"
        if isinstance(vector_store_type, str):
            vector_store_type = getattr(
                SupportedVectorStoreTypes, vector_store_type.upper()
            )
        if vector_store_type == SupportedVectorStoreTypes.MILVUS:
            from rag.server.kb.milvus_kb_service import MilvusKBService

            # dummy code
            return MilvusKBService(kb_name)
