from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class KBDatabaseInterface(ABC):
    """
    定义了KB和MongoDB交互的接口

    """

    @staticmethod
    @abstractmethod
    def connect_to_db():
        """
        静态连接数据库
        """
        pass

    @staticmethod
    @abstractmethod
    def insert_document_with_uuid(db_name: str, collection_name: str, document: Dict[str, Any]) -> str:
        """
        将单个文档插入指定集合

        :param db_name: 数据库名
        :param collection_name: Mongodb集合名
        :param document: 要插入的文档
        :return: 插入文档的ID
        """
        pass

    @staticmethod
    @abstractmethod
    def insert_document(db_name: str, collection_name: str, documents: List[Dict[str, Any]]) -> int:
        """
        将多个数据插入到指定collection

        :param db_name: 数据库名
        :param collection_name: 集合名
        :param documents: 多个要插入的数据
        :return: 插入成功的数量
        """
        pass

    @staticmethod
    @abstractmethod
    def insert_json_data_with_uuid(db_name: str, collection_name: str, json_file: str):
        """
        读取JSON文件并为每个文档添加UUID，然后将其数据插入到MongoDB中

        :param db_name: 数据库名称
        :param collection_name: 集合名称
        :param json_file: JSON文件路径
        :return: 所有插入文档的UUID列表
        """
        pass

    @staticmethod
    @abstractmethod
    def update_document(db_name: str, collection_name: str, uuid_str: str, update_fields: Dict[str, Any]) -> int:
        """
        更新指定uuid的文档

        :param db_name: 数据库名称
        :param collection_name: 集合名称
        :param uuid_str: 用于查找文档的uuid
        :param update_fields: 要更新的字段及其新值（字典）
        :return: 更新的文档数
        """
        pass

    @staticmethod
    @abstractmethod
    def update_documents(db_name: str, collection_name: str, query: dict, update_fields: Dict[str, Any]) -> int:
        """
        更新多个文档（根据查询条件）

        :param db_name: 数据库名称
        :param collection_name: 集合名称
        :param query: 查询条件
        :param update_fields: 要更新的字段及其新值（字典）
        :return: 更新的文档数
        """
        pass

    @staticmethod
    @abstractmethod
    def delete_document(db_name: str, collection_name: str, uuid_str: str) -> int:
        """
        删除指定uuid的文档

        :param db_name: 数据库名称
        :param collection_name: 集合名称
        :param uuid_str: 用于查找文档的uuid
        :return: 删除的文档数量
        """
        pass

    @staticmethod
    @abstractmethod
    def delete_documents_by_list(db_name: str, collection_name: str, documents: List[Dict[str, Any]]) -> int:
        """
        通过具体数据来删除数据 -> 针对kb db之间的数据

        :param db_name: 数据库名
        :param collection_name: 表名
        :param documents: 要删除的数据集
        :return: 删除的记录数量
        """
        pass

    @staticmethod
    @abstractmethod
    def delete_documents(db_name: str, collection_name: str, query: Dict[str, Any]) -> int:
        """
        删除多个文档（根据查询条件）

        :param db_name: 数据库名称
        :param collection_name: 集合名称
        :param query: 查询条件
        :return: 删除的文档数量
        """
        pass

    @staticmethod
    @abstractmethod
    def find_document_by_uuid(db_name: str, collection_name: str, uuid_str: str) -> Dict[str, Any]:
        """
        通过uuid查询文档

        :param db_name: 数据库名称
        :param collection_name: Mongodb集合名
        :param uuid_str: 查找的uuid
        :return: 文档List结果
        """
        pass

    @staticmethod
    @abstractmethod
    def find_document(db_name: str, collection_name: str, query: Dict[str, Any],
                      projection: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
        """
        查询指定条件的内容

        :param db_name: 数据库名
        :param collection_name: collection名
        :param query: 查询条件
        :param projection: 要在返回的文档中包含或排除的字段
        :return: 要查找的文档
        """
        pass

    @staticmethod
    @abstractmethod
    def aggregate_documents(db_name: str, collection_name: str, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        执行聚合操作

        :param db_name: 数据库名称
        :param collection_name: 集合名称
        :param pipeline: 聚合管道（列表）
        :return: 聚合结果
        """
        pass
