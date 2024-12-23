from typing import List

SPLIT_TOKEN = "_0_"


def map_collection_name(kb_name: str, collection_name: str) -> str:
    kb_name = kb_name.strip()
    collection_name = collection_name.strip()
    if SPLIT_TOKEN in kb_name or len(kb_name) == 0:
        raise ValueError(f"Invalid kb_name, should not contain {SPLIT_TOKEN}")
    if SPLIT_TOKEN in collection_name or len(collection_name) == 0:
        raise ValueError(f"Invalid collection_name, should not contain {SPLIT_TOKEN}")
    return f"{kb_name}{SPLIT_TOKEN}{collection_name}"


def unmap_collection_name(collection_name: str) -> List[str]:
    return collection_name.split(SPLIT_TOKEN)


def filter_collection_with_kb_name(kb_name: str, collections: List[str]) -> List[str]:
    if SPLIT_TOKEN in kb_name or len(kb_name) == 0:
        raise ValueError("Invalid kb_name")
    collection_names = []
    for collection_name in collections:
        collection_kb_name, collection_name = unmap_collection_name(collection_name)
        if kb_name == collection_kb_name:
            collection_names.append(collection_name)
    return collection_names
