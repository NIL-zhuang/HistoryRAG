from typing import Dict, List, Union
from rag.server.models.kb_spec import Context
from rag.server.models.model_spec import History



def construct_message(
    query: str,
    history: List[History],
    contexts: List[Context],
    prompt_template: Union[str, Dict[str, str], List[Dict[str, str]]]
) -> List[Dict[str, str]]:

    prompt_template[0]["content"] = prompt_template[0]["content"].format(**locals())
    prompt_template[1]["content"] = prompt_template[1]["content"].format(**locals())

    return prompt_template
