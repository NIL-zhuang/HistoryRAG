from typing import Dict, List, Union

from rag.server.specs import Context, History
from rag.settings import Settings


def construct_message(
    query: str,
    history: List[History],
    contexts: List[Context],
    prompt_template: Union[str, Dict[str, str], List[Dict[str, str]]],
) -> List[Dict[str, str]]:
    # TODO: Implement this function
    return [
        {"role": "system", "content": Settings.prompt_settings.DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
