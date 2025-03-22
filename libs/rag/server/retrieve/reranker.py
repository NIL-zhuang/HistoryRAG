from typing import List

from rag.server.models.kb_spec import Context

'''
基于rules的重排序算法
简单来说，一个文档在越多个召回结果中排名靠前，则总排行越靠前
k为平滑参数，一般情况下使用默认值60即可
'''
def rrf(doc_lists: List[List[Context]], k=60):
    # 收集所有唯一元素
    elements = set()
    for doc_list in doc_lists:
        elements.update(doc_list)
    elements = list(elements)  # 转换为列表以便排序

    # 为每个列表创建元素到排名的字典（排名从1开始）
    list_rank_dicts = []
    for doc_list in doc_lists:
        rank_dict = {element: i + 1 for i, element in enumerate(doc_list)}
        list_rank_dicts.append(rank_dict)

    # 计算每个元素的RRF总分
    scores = {}
    for element in elements:
        total = 0.0
        for rank_dict in list_rank_dicts:
            if element in rank_dict:
                rank = rank_dict[element]
                total += 1 / (k + rank)
        scores[element] = total

    # 按总分降序排序并返回结果
    sorted_elements = sorted(elements, key=lambda x: scores[x], reverse=True)
    return sorted_elements