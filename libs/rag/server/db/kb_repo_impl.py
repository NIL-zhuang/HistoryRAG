from abc import ABC

from rag.server.db.kb_repo import KBDatabaseInterface
import uuid
import json
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from rag.settings import Settings


class MongoDB(KBDatabaseInterface, ABC) :
    db_client = None

    @staticmethod
    def connect_to_db():
        if Settings.db_settings.DB_USERNAME is None or Settings.db_settings.DB_USERNAME == "":
            MongoDB.db_client = MongoClient(f"mongodb://{Settings.db_settings.DB_HOST}:{Settings.db_settings.DB_PORT}")
        else:
            MongoDB.db_client = MongoClient(
                f"mongodb://{Settings.db_settings.DB_USERNAME}:{Settings.db_settings.DB_PASSWORD}@{Settings.db_settings.DB_HOST}:{Settings.db_settings.DB_PORT}")

    @staticmethod
    def insert_document_with_uuid(db_name: str, collection_name: str, document: Dict[str, Any]) -> str:
        # 生成一个新的 UUID
        document["_id"] = str(uuid.uuid4())

        # 获取数据库和集合
        db = MongoDB.db_client[db_name]
        collection = db[collection_name]

        # 插入文档
        result = collection.insert_one(document)
        return str(result.inserted_id)  # 返回插入的文档ID（UUID）

    @staticmethod
    def insert_document(db_name: str, collection_name: str, documents: List[Dict[str, Any]]) -> int:
        for document in documents:
            # 生成新的 UUID
            document["_id"] = str(uuid.uuid4())

        # 获取数据库和集合
        db = MongoDB.db_client[db_name]
        collection = db[collection_name]

        # 插入文档
        result = collection.insert_many(documents)

        return len(result.inserted_ids)

    @staticmethod
    def insert_json_data_with_uuid(db_name: str, collection_name: str, json_file: str):
        # 读取JSON文件
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)  # 将文件中的内容解析为Python对象

        # 为每个文档添加UUID
        for item in data:
            # 生成一个UUID并作为_id插入
            item["_id"] = str(uuid.uuid4())  # 将UUID作为_id字段或自定义字段

        db = MongoDB.db_client[db_name]
        collection = db[collection_name]

        # 批量插入数据到MongoDB
        result = collection.insert_many(data)

        # 获取所有插入文档的UUID
        inserted_uuids = list(result.inserted_ids)

        return inserted_uuids  # 返回插入文档的UUID列表

    @staticmethod
    def update_document(db_name: str, collection_name: str, uuid_str: str, update_fields: Dict[str, Any]) -> int:
        db = MongoDB.db_client[db_name]
        collection = db[collection_name]

        # 使用 $set 更新字段
        result = collection.update_one(
            {"_id": uuid_str}, {"$set": update_fields}  # 查询条件，根据 _id 查找  # 更新操作
        )
        return result.modified_count  # 返回修改的文档数量

    @staticmethod
    def update_documents(db_name: str, collection_name: str, query: dict, update_fields: Dict[str, Any]) -> int:
        db = MongoDB.db_client[db_name]
        collection = db[collection_name]

        # 使用 $set 更新多个文档
        result = collection.update_many(query, {"$set": update_fields})  # 查询条件  # 更新操作
        return result.modified_count  # 返回修改的文档数量

    @staticmethod
    def delete_document(db_name: str, collection_name: str, uuid_str: str) -> int:
        db = MongoDB.db_client[db_name]
        collection = db[collection_name]

        # 根据UUID删除文档
        result = collection.delete_one({"_id": uuid_str})
        return result.deleted_count  # 返回删除的文档数量

    @staticmethod
    def delete_documents_by_list(db_name: str, collection_name: str, documents: List[Dict[str, Any]]) -> int:
        db = MongoDB.db_client[db_name]
        collection = db[collection_name]

        deleted_count = 0
        for document in documents:
            result = collection.delete_many(document)
            deleted_count += result.deleted_count
        return deleted_count  # 返回删除的文档数量

    @staticmethod
    def delete_documents(db_name: str, collection_name: str, query: Dict[str, Any]) -> int:
        db = MongoDB.db_client[db_name]
        collection = db[collection_name]

        # 根据查询条件删除多个文档
        result = collection.delete_many(query)
        return result.deleted_count  # 返回删除的文档数量

    @staticmethod
    def find_document_by_uuid(db_name: str, collection_name: str, uuid_str: str) -> Dict[str, Any]:
        db = MongoDB.db_client[db_name]
        collection = db[collection_name]

        result = collection.find_one({"_id": uuid_str})
        return result  # 返回匹配的文档

    @staticmethod
    def find_document(db_name: str, collection_name: str, query: Dict[str, Any],
                      projection: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
        db = MongoDB.db_client[db_name]
        collection = db[collection_name]
        cursor = collection.find(query, projection)
        return list(cursor)

    @staticmethod
    def aggregate_documents(db_name: str, collection_name: str, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        db = MongoDB.db_client[db_name]
        collection = db[collection_name]

        # 执行聚合操作
        result = collection.aggregate(pipeline)
        return list(result)  # 返回聚合结果的列表

if __name__ == '__main__':
    MongoDB.connect_to_db()
    list = MongoDB.find_document("KnowledgeBase", "mycollection", {"user_name": "adamlee"})
    for doc in list:
        print(doc)
