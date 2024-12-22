from typing import Any, Dict, List

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, MilvusClient
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
        embed_dim = embedding_model_config.meta_data.get("embed_size", 1024)
        self.embed_func = PlatformLLM(embedding_model_config).embed
        if kb_info is None:
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
                max_length=5120,
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
            output_fields=["metadata", "content"],
        )[0]
        contexts = [Context.model_validate(obj["entity"]) for obj in results]
        return contexts

    def do_init(
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

    def load_collection(self, collection_name: str):
        self.client.load_collection(collection_name)

    def release_collection(self, collection_name: str):
        self.client.release_collection(collection_name)

    def do_add_context(self, collection_name: str, context: Context):
        context_data = context.model_dump()
        embedding = self.embed_func(context.content)
        context_data["embedding"] = embedding
        self.client.insert(collection_name=collection_name, data=context_data)

    def get_obj_by_uuid(self, collection_name: str, uuid: str):
        return self.client(collection_name=collection_name, ids=[uuid])


if __name__ == "__main__":
    kb = MilvusKBService("default")

    collection_name = "history_rag"
    kb.client.drop_collection(collection_name=collection_name)

    kb.do_init(collection_name=collection_name)
    for i in range(10):
        context = Context(metadata={}, content=f"test test test content_{i}")
        kb.do_add_context(context=context, collection_name=collection_name)

    context = kb.search(query="test 0", collection_name=collection_name, top_k=5)
    for con in context:
        print(con)
