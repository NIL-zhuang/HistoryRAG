from typing import Dict, List, Union
from rag.server.models.kb_spec import Context
from rag.server.models.model_spec import History

USER_PROMPT_INDEX = 1

def construct_message(
    query: str,
    history: List[History],
    contexts: List[Context],
    prompt_template: Union[str, Dict[str, str], List[Dict[str, str]]]
) -> List[Dict[str, str]]:
    format_context = {
        "query": query,
        "history": history,
        "contexts": contexts,
    }

    prompt_template[USER_PROMPT_INDEX]["content"] = prompt_template[USER_PROMPT_INDEX]["content"].format(format_context)

    return prompt_template
