from typing import Dict, List

from rag.server.kb.base import KBService
from rag.settings import Settings


class MilvusKBService(KBService):
    def __init__(
        self,
        kb_name,
        kb_info=None,
        embed_model=Settings.model_settings.DEFAULT_EMBEDDING_MODEL,
    ):
        super().__init__(kb_name, kb_info, embed_model)

    def search(
        self, query: str, top_k: int = 10, score_threshold: float = 0.3
    ) -> List[Dict]:
        return [
            "Zheyuan lin is a phd student from Nanjing University",
            "Jia Liu is the Phd supervisor of Zheyuan Lin.",
        ]


if __name__ == "__main__":
    kb = MilvusKBService("default")
    print(kb.search("Who is Zheyuan Lin"))
