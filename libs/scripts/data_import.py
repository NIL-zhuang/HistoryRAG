# ENSURE YOU HAVE RUN THE HISTORY_RAG APP

import argparse
import json
from pathlib import Path
from typing import Dict, List

import requests
from tqdm import trange

# https://milvus.io/docs/string.md
MAX_CONTENT_LENGTH = 64000


def import_data(data: List[Dict], url: str, kb_name: str, collection_name: str):
    info_list = []
    for d in data:
        info = {
            "metadata": {
                "series_name": d["series_name"],
                "file_name": d["file_name"],
                "title": d["title"],
                "start_page": d["start_page"],
                "end_page": d["end_page"],
            },
            "content": d["full_text"][:MAX_CONTENT_LENGTH],
        }
        info_list.append(info)

    payload = {
        "kb_name": kb_name,
        "collection_name": collection_name,
        "context": info_list,
    }
    response = requests.post(f"{url}/kb/add_context", json=payload)
    return response


def load_data(fpath: Path, limit: int = 0):
    with open(fpath, "r", encoding="utf-8") as f:
        if fpath.suffix == ".jsonl":
            if limit == 0:
                lines = f.readlines()
            else:
                lines = [f.readline() for _ in range(limit)]
            data = [json.loads(line) for line in lines]
        elif fpath.suffix == ".json":
            data = json.load(f)
            if limit > 0:
                data = data[:limit]
    return data


def main(args: argparse.Namespace):
    data = load_data(args.data_path, args.limit)
    for i in trange(0, len(data), args.batch_size):
        batch = data[i : i + args.batch_size]
        response = import_data(
            batch, args.server_url, args.kb_name, args.collection_name
        )
        print(response)


def parse_args():
    parser = argparse.ArgumentParser(description="Import data from the history_rag app")
    parser.add_argument("--data_path", type=Path, help="Path to the jsonl input file")
    parser.add_argument("--server_url", type=str, help="URL of the RAG server")
    parser.add_argument("--kb_name", type=str, help="Knowledge base name")
    parser.add_argument("--collection_name", type=str, help="Collection name")
    parser.add_argument(
        "--limit", type=int, help="Number of examples to import", default=0
    )
    parser.add_argument("--batch_size", type=int, help="Batch size", default=16)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opts = parse_args()
    main(opts)
