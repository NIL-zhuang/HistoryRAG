from abc import ABC, abstractmethod
import uuid
import json


class KBDatabaseInterface(ABC):
    """
    定义了KB和MongoDB交互的接口

    """

    @abstractmethod
    def insert_document_with_uuid(self, db_name: str, collection_name: str, document: dict) -> str:
        """
        将单个文档插入指定集合

        :param db_name: 数据库名
        :param collection_name: Mongodb集合名
        :param document: 要插入的文档
        :return: 插入文档的ID
        """
        pass

    @abstractmethod
    def insert_json_data_with_uuid(self, db_name: str, collection_name: str, json_file: str):
        """
        读取JSON文件并为每个文档添加UUID，然后将其数据插入到MongoDB中

        :param db_name: 数据库名称
        :param collection_name: 集合名称
        :param json_file: JSON文件路径
        :return: 所有插入文档的UUID列表
        """
        pass

    @abstractmethod
    def update_document(self, db_name: str, collection_name: str, uuid_str: str, update_fields: dict) -> int:
        """
        更新指定uuid的文档

        :param db_name: 数据库名称
        :param collection_name: 集合名称
        :param uuid_str: 用于查找文档的uuid
        :param update_fields: 要更新的字段及其新值（字典）
        :return: 更新的文档数
        """
        pass

    @abstractmethod
    def update_documents(self, db_name: str, collection_name: str, query: dict, update_fields: dict) -> int:
        """
        更新多个文档（根据查询条件）

        :param db_name: 数据库名称
        :param collection_name: 集合名称
        :param query: 查询条件
        :param update_fields: 要更新的字段及其新值（字典）
        :return: 更新的文档数
        """
        pass

    @abstractmethod
    def delete_document(self, db_name: str, collection_name: str, uuid_str: str) -> int:
        """
        删除指定uuid的文档

        :param db_name: 数据库名称
        :param collection_name: 集合名称
        :param uuid_str: 用于查找文档的uuid
        :return: 删除的文档数量
        """
        pass

    @abstractmethod
    def delete_documents(self, db_name: str, collection_name: str, query: dict) -> int:
        """
        删除多个文档（根据查询条件）

        :param db_name: 数据库名称
        :param collection_name: 集合名称
        :param query: 查询条件
        :return: 删除的文档数量
        """
        pass

    @abstractmethod
    def find_document_by_uuid(self, db_name: str, collection_name: str, uuid_str: str) -> dict:
        """
        查询文档

        :param db_name: 数据库名称
        :param collection_name: Mongodb集合名
        :param query: 要查询的文档
        :param projection: 筛选条件
        :return: 文档List结果
        """
        pass

    @abstractmethod
    def aggregate_documents(self, db_name: str, collection_name: str, pipeline: list) -> list:
        """
        执行聚合操作

        :param db_name: 数据库名称
        :param collection_name: 集合名称
        :param pipeline: 聚合管道（列表）
        :return: 聚合结果
        """
        pass


class MongoDB(KBDatabaseInterface):
    def __init__(self, db_client):
        """
        用MongoDB客户端实例初始化MongoDBInterface

        :param db_client: MongoDB客户端实例
        """
        self.db_client = db_client

    def insert_document_with_uuid(self, db_name: str, collection_name: str, document: dict) -> str:
        # 生成一个新的 UUID
        document["_id"] = str(uuid.uuid4())

        # 获取数据库和集合
        db = self.db_client[db_name]
        collection = db[collection_name]

        # 插入文档
        result = collection.insert_one(document)
        return str(result.inserted_id)  # 返回插入的文档ID（UUID）

    def insert_json_data_with_uuid(self, db_name: str, collection_name: str, json_file: str):
        # 读取JSON文件
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)  # 将文件中的内容解析为Python对象

        # 为每个文档添加UUID
        for item in data:
            # 生成一个UUID并作为_id插入
            item["_id"] = str(uuid.uuid4())  # 将UUID作为_id字段或自定义字段

        db = self.db_client[db_name]
        collection = db[collection_name]

        # 批量插入数据到MongoDB
        result = collection.insert_many(data)

        # 获取所有插入文档的UUID
        inserted_uuids = list(result.inserted_ids)

        return inserted_uuids  # 返回插入文档的UUID列表

    def update_document(self, db_name: str, collection_name: str, uuid_str: str, update_fields: dict) -> int:
        db = self.db_client[db_name]
        collection = db[collection_name]

        # 使用 $set 更新字段
        result = collection.update_one(
            {"_id": uuid_str}, {"$set": update_fields}  # 查询条件，根据 _id 查找  # 更新操作
        )
        return result.modified_count  # 返回修改的文档数量

    def update_documents(self, db_name: str, collection_name: str, query: dict, update_fields: dict) -> int:
        db = self.db_client[db_name]
        collection = db[collection_name]

        # 使用 $set 更新多个文档
        result = collection.update_many(query, {"$set": update_fields})  # 查询条件  # 更新操作
        return result.modified_count  # 返回修改的文档数量

    def delete_document(self, db_name: str, collection_name: str, uuid_str: str) -> int:
        db = self.db_client[db_name]
        collection = db[collection_name]

        # 根据UUID删除文档
        result = collection.delete_one({"_id": uuid_str})
        return result.deleted_count  # 返回删除的文档数量

    def delete_documents(self, db_name: str, collection_name: str, query: dict) -> int:
        db = self.db_client[db_name]
        collection = db[collection_name]

        # 根据查询条件删除多个文档
        result = collection.delete_many(query)
        return result.deleted_count  # 返回删除的文档数量

    def find_document_by_uuid(self, db_name: str, collection_name: str, uuid_str: str) -> dict:
        db = self.db_client[db_name]
        collection = db[collection_name]

        result = collection.find_one({"_id": uuid_str})
        return result  # 返回匹配的文档

    def aggregate_documents(self, db_name: str, collection_name: str, pipeline: list) -> list:
        db = self.db_client[db_name]
        collection = db[collection_name]

        # 执行聚合操作
        result = collection.aggregate(pipeline)
        return list(result)  # 返回聚合结果的列表
