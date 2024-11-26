from fastapi import APIRouter, Request
import os
import sys
import json
from sentence_transformers import SentenceTransformer
from defog import Defog
import numpy as np
from sqlcoder.faiss_manager import FaissManager
from sqlcoder.metadata_utils import (
    detect_device_type,
    convert_nested_dict_to_list,
    get_metadata_json,
    generate_cloumn_metadata_json,
    generate_table_metadata_json,
    get_column_to_ddl,
    get_table_to_ddl,
    convert_metadata_to_ddl,
)


#向量索引路径
index_path = "./index" 

home_dir = os.path.expanduser("~")
defog_path = os.path.join(home_dir, ".defog")


# 检测设备类型
device_type = detect_device_type()


# 加载量化DDL描述模型
def load_ddl_vector_model(device_type):
    """
    根据设备类型加载适合的DDL向量化模型。
    :param device_type: 设备类型 ('gpu', 'cpu', 'apple_silicon')。
    :return: 加载的模型实例。
    """
    if device_type == "gpu":
        print("加载GPU优化的DDL模型 WangZeJun/simbert-base-chinese...")
        return SentenceTransformer('WangZeJun/simbert-base-chinese', device='cuda')  # 更强大但资源需求更高的模型
    elif device_type == "apple_silicon":
        print("加载适用于 Apple Silicon 的DDL模型 paraphrase-MiniLM-L6-v2...")
        return SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')  # Apple Silicon 优化
    else:
        print("加载轻量化的DDL模型 paraphrase-MiniLM-L6-v2 (CPU 模式)...")
        return SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')  # 轻量化模型


# 加载DDL向量化模型
ddl_vector_model = load_ddl_vector_model(device_type)




# 提供DDL向量化功能
def vectorize_ddl(ddl_text: str):
    """
    将DDL描述文本向量化。
    :param ddl_text: DDL描述字符串。
    :return: 向量化结果 (numpy.ndarray)。
    """
    if not ddl_vector_model:
        raise ValueError("DDL向量化模型未加载！")
    return ddl_vector_model.encode(ddl_text)


#通过传入的DDL描述向量化，转化成向量主键二维数组
def vectorize_ddl_save_ids(ddl_texts):
    if not ddl_texts:
        return {"error": "No ddl_texts provided"}

    print(f"正在向量化文本: {ddl_texts}")
    
    # 创建空列表存储所有的向量和向量ID
    all_vectors = []
    all_vector_ids = []
    
    for ddl_text in ddl_texts:
        # 跳过空值或空字符串
        if not ddl_text.get('table_description') or ddl_text.get('table_description').strip() == "":
            continue

        # 向量化每一个 ddl_text
        vector = vectorize_ddl(ddl_text.get('table_description'))
        
        # 保存向量到 FAISS 并获取 ID
        vector_ids = save_vector(vector,ddl_text.get('table_description'))
        
        # 将 NumPy 数组转换为 Python 列表
        vector_list = vector.tolist()
        vector_ids_list = vector_ids.tolist()
        
        # 添加到结果列表
        all_vectors.append(vector_list)
        all_vector_ids.append(vector_ids_list)
        # 返回的格式类似于 {"vector_ids":[[0],[1]],"vector":[[0.07723584026098252,-0.06650960445404053,-0.07342207431793213],[0.07723584026098252,-0.06650960445404054,-0.07342207431793216]]}
        # 返回的是二维数组
    return {"vector_ids":all_vector_ids}

def save_vector(vectors: np.ndarray,value: str):
    defog = Defog()
    db_creds = defog.db_creds
    database_name = db_creds['database']  # 数据库名 

    faiss_manager = FaissManager(base_dir=index_path, dim=768)

    vector_ids = faiss_manager.add_vectors(database_name, vectors, value)
    print(f"添加向量后返回的ID:")
    print(vector_ids)
    return vector_ids




def search_vectorize(search_text: str, top_k: int = 5, distance_threshold: float = 0.5):
    
    if not search_text:
        return {"error": "No search_text provided"}
    print(f"待查找的值: {search_text}")

    # 设置返回数量，默认为5
    if not top_k:
        top_k = 5

    # 设定距离阈值,默认是0.5
    if not distance_threshold:
        distance_threshold = 0.5


    # 向量化一个 search_text
    vector = vectorize_ddl(search_text)
    
    defog = Defog()
    db_creds = defog.db_creds
    database_name = db_creds['database']  # 数据库名 
    #top_k = rtn_num  # 返回前 5 个最近邻

    faiss_manager = FaissManager(base_dir=index_path, dim=768)
    # 调用 search 方法进行查找
    search_results = faiss_manager.search(database_name, vector, top_k)
    #查找结果: {'indices': [13, 14, 240, 245, 5], 'distances': [0.208731546998024, 0.2659868001937866, 0.31992244720458984, 0.3447423279285431, 0.36340105533599854]}
    print(f"查找结果: {search_results}")
    indices = search_results.get("indices")  # 索引 ID
    distances = search_results.get("distances")  # 距离值
    
    # 按照距离过滤结果
    filtered_results = [
        (idx, dist) for idx, dist in zip(indices, distances) 
        if 0 <= dist <= distance_threshold
    ]
    
    # 拆分过滤后的索引和距离
    filtered_indices = [item[0] for item in filtered_results]
    filtered_distances = [item[1] for item in filtered_results]

    print(f"过滤后的最近邻索引ID: {filtered_indices}")
    print(f"过滤后的距离: {filtered_distances}")
    
    return {
        "indices": filtered_indices,
        "distances": filtered_distances
    }



def search_vectorize_join_ddl(search_text: str, top_k: int = 5, distance_threshold: float = 0.5):
    #获取查找值
    search_vectorize_value = search_vectorize(search_text, top_k, distance_threshold)
    #获取元数据json表格
    try:
        defog = Defog()
    except:
        return {"error": "no defog instance found"}

    try:
        with open(os.path.join(defog_path, "metadata_vector.json"), "r") as f:
            table_ddl_json = json.load(f)
    except:
        table_ddl_json = []

    print(f"元数据json表格: {table_ddl_json}")

    # 获取查询到的 indices 和 distances
    indices = search_vectorize_value.get("indices", [])
    distances = search_vectorize_value.get("distances", [])
    
    if not indices or not distances:
        return {"error": "No valid indices or distances found in search results"}
    
    # 过滤 table_ddl_json 数据，保留 vector_ids 在 indices 中的项
    filtered_data = [
        item for item in table_ddl_json.get("ddl_texts", [])
        if item["vector_ids"] in indices
    ]
    
    # 构建返回的 table_name 和 table_description 列表
    table_names = [item["table_name"] for item in filtered_data]
    table_descriptions = [item["table_description"] for item in filtered_data]
    

    metadata= get_metadata_json(table_names)

    ddl = convert_metadata_to_ddl(metadata)

    # 返回结果
    return {
        "indices": indices,
        "distances": distances,
        "table_names": table_names,
        "table_descriptions": table_descriptions,
        "metadata":metadata,
        "ddl":ddl
    }

