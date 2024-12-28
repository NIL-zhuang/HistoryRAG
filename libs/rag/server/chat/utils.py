from symtable import Function
from typing import Dict, List, Union
from typing import Callable
from rag.server.models.kb_spec import Context
from rag.server.models.model_spec import History
from rag.settings import Settings



def construct_message(
    query: str,
    history: List[History],
    contexts: List[Context],
    prompt_template: Callable[[List[Context], str, List[History]], List[Dict[str, str]]]
) -> List[Dict[str, str]]:

    return prompt_template(contexts, query, history)
