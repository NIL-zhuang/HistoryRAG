from functools import partial
from typing import Any, Dict, List, Union

from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient, Function, FunctionType
from rag.server.kb.base import KBService
from rag.server.llm.proxy_llm import PlatformLLM
from rag.server.llm.utils import get_model_configs
from rag.server.models.kb_spec import Context, ContextMetadata
from rag.server.models.model_spec import ModelConfig
from rag.settings import Settings


class MilvusKBService(KBService):

    def __init__(
        self,
        kb_name: str = "default",
        kb_info: str = None,
        embed_model: str = Settings.model_settings.DEFAULT_EMBEDDING_MODEL,
    ):
        self.client = MilvusClient(
            uri=Settings.kb_settings.MILVUS_HOST,
            token=Settings.kb_settings.MILVUS_TOKEN,
        )

        embedding_model_config: ModelConfig = get_model_configs(embed_model)
        embed_dim = embedding_model_config.meta_data.get(
            "embed_size", Settings.model_settings.DEFAULT_EMBEDDING_SIZE
        )
        self.context_window = embedding_model_config.meta_data.get(
            "context_window", Settings.model_settings.DEFAULT_EMBEDDING_CONTEXT_WINDOW
        )
        self.embed_func = partial(
            PlatformLLM(embedding_model_config).embed,
            context_window=self.context_window,
        )
        self.bm25_func = Function(
            name="text_bm25_emb",  # Function name
            input_field_names=["content"],  # Name of the VARCHAR field containing raw text data
            output_field_names=["sparse"],
            # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
            function_type=FunctionType.BM25,
        )
        if kb_info is None or len(kb_info.strip()) == 0:
            kb_info = f"Milvus KB Service, based on {embed_model}, dim {embed_dim}"
        super().__init__(kb_name, kb_info, embed_model)
        self._DEFAULT_FIELDS = [
            FieldSchema(
                name="uuid",
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=True,
                max_length=36,
            ),
            FieldSchema(
                name="metadata", dtype=DataType.JSON, is_primary=False, nullable=True
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                is_primary=False,
                max_length=65535,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                is_primary=False,
                dim=embed_dim,
            ),
        ]
        self._DEFAULT_INDEX_PARAMS = [
            {
                "field_name": "embedding",
                "index_type": "AUTOINDEX",
                "metric_type": "COSINE",
            }
        ]
        # 用于稀疏检索的Field和Index
        self._SPARSE_FIELDS = [
            FieldSchema(
                name="uuid",
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=True,
                max_length=36,
            ),
            FieldSchema(
                name="metadata", dtype=DataType.JSON, is_primary=False, nullable=True
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                enable_analyzer=True,
                analyzer_params={"type": "chinese"},
                is_primary=False,
                max_length=65535,
            ),
            FieldSchema(
                name="sparse",
                dtype=DataType.SPARSE_FLOAT_VECTOR,
                is_primary=False,
            )
        ]
        self._SPARSE_INDEX_PARAMS = self.client.prepare_index_params().add_index(
            field_name="sparse",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",  # Inverted index type for sparse vectors
            metric_type="BM25",
            params={"inverted_index_algo": "DAAT_MAXSCORE"},
            # Algorithm for building and querying the index. Valid values: DAAT_MAXSCORE, DAAT_WAND, TAAT_NAIVE.
        )

    def list_collection(self):
        return self.client.list_collections()

    def search(
        self,
        query: str,
        collection_name: str,
        top_k: int = Settings.kb_settings.VS_TOP_K,
        score_threshold: float = Settings.kb_settings.SCORE_THRESHOLD,
        search_params: Dict[str, Any] = None,
    ) -> List[Context]:
        if search_params is None:
            search_params = {"metric_type": "COSINE"}
        query_embedding = self.embed_func(query)
        results = self.client.search(
            collection_name=collection_name,
            anns_field="embedding",
            data=[query_embedding],
            search_params=search_params,
            limit=top_k,
            output_fields=["id", "distance", "metadata", "content"],
        )[0]
        results = [r for r in results if r["distance"] >= score_threshold]
        results = [
            {"id": r["id"], "distance": r["distance"], **r["entity"]} for r in results
        ]
        contexts = [Context.model_validate(obj) for obj in results]
        return contexts

    def create_collection(
        self,
        collection_name: str,
        collection_info: str = "",
        fields: List[FieldSchema] = None,
        index_params: List[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        {"state": "<LoadState: Loaded>"}
        """
        if fields is None:
            fields = self._DEFAULT_FIELDS
        if index_params is None and "embedding" in [field.name for field in fields]:
            index_params = self._DEFAULT_INDEX_PARAMS
        schema = CollectionSchema(
            fields=fields, description=collection_info, enable_dynamic_field=True
        )
        if index_params is not None:
            index_parameters = self.client.prepare_index_params()
            for param in index_params:
                index_parameters.add_index(**param)
        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_parameters,
        )
        load_state = self.client.get_load_state(collection_name)
        return load_state

    def create_collection_for_sparse(
        self,
        collection_name: str,
        collection_info: str = "",
        fields: List[FieldSchema] = None,
        index_params: List[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        if fields is None:
            fields = self._SPARSE_FIELDS
        schema = CollectionSchema(
            fields=fields, description=collection_info, enable_dynamic_field=True
        )
        if index_params is None and "sparse" in [field.name for field in fields]:
            index_params = self._SPARSE_INDEX_PARAMS
            schema.add_function(self.bm25_func)
        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        load_state = self.client.get_load_state(collection_name)
        return load_state

    def load_collection(self, collection_name: str):
        self.client.load_collection(collection_name)

    def release_collection(self, collection_name: str):
        self.client.release_collection(collection_name)

    def drop_collection(self, collection_name):
        if collection_name in self.list_collection():
            self.client.drop_collection(collection_name)

    def add_context(self, collection_name: str, context: Union[Context, List[Context]]):
        if isinstance(context, Context):
            context = [context]
        context_data = [c.model_dump() for c in context]
        content = [c["content"] for c in context_data]
        embedding = self.embed_func(content)
        for c, embed in zip(context_data, embedding):
            c["embedding"] = embed
        self.client.insert(collection_name=collection_name, data=context_data)

    def get_obj_by_uuid(self, collection_name: str, uuid: str):
        return self.client(collection_name=collection_name, ids=[uuid])

    def save_vector_store(self):
        pass

    def add_context_for_sparse(self, collection_name: str, context: Union[Context, List[Context]]):
        if isinstance(context, Context):
            context = [context]
        context_data = [c.model_dump() for c in context]
        self.client.insert(collection_name=collection_name, data=context_data)

    def full_text_search(
            self,
            keywords: List[str],
            collection_name: str,
            top_k: int = Settings.kb_settings.VS_TOP_K,
            score_threshold: float = Settings.kb_settings.SCORE_THRESHOLD,
            search_params: Dict[str, Any] = None,
    ) -> List[Context]:
        if search_params is None:
            search_params = {'params': {'drop_ratio_search': score_threshold}}
        results = self.client.search(
            collection_name=collection_name,
            anns_field="sparse",
            data=keywords,
            search_params=search_params,
            limit=top_k,
            output_fields=["id", "metadata", "content"],
        )[0]
        results = [
            {"id": r["id"], "distance": 1.0, **r["entity"]} for r in results
        ]
        contexts = [Context.model_validate(obj) for obj in results]
        return contexts


if __name__ == "__main__":
    kb = MilvusKBService("default")

    collection_name = "history_rag"
    kb.drop_collection(collection_name=collection_name)
    kb.create_collection(collection_name=collection_name)
    context = [
        Context(metadata={}, content=f"test test test content_{i}") for i in range(10)
    ]
    kb.add_context(context=context, collection_name=collection_name)

    context = kb.search(query="test 0", collection_name=collection_name, top_k=5)
    for con in context:
        print(con)
