import json
from typing import Dict, List, Union, Any

from rag.server.llm.base import LLM
from rag.server.models.kb_spec import Context
from rag.server.models.model_spec import History
from rag.settings import Settings

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

    prompt_template[USER_PROMPT_INDEX]["content"] = prompt_template[USER_PROMPT_INDEX]["content"].format(**format_context)

    return prompt_template

def rewrite_query(
    query: str,
    llm: LLM,
    rewrite_template: Union[str, Dict[str, str], List[Dict[str, str]]]
) -> Dict[str, Any]:
    format_context = {
        "query": query,
        "REWRITE_QUERIES_NUM" : Settings.prompt_settings.REWRITE_QUERIES_NUM
    }
    rewrite_template[USER_PROMPT_INDEX]["content"] = rewrite_template[USER_PROMPT_INDEX]["content"].format(**format_context)
    result = llm.chat(rewrite_template)
    result = json.loads(result)
    return result