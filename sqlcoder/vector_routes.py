from fastapi import APIRouter, Request
import os
import sys
from sentence_transformers import SentenceTransformer
from defog import Defog
import numpy as np

from sqlcoder.faiss_manager import FaissManager
from sqlcoder.query_routes import detect_device_type


router = APIRouter()


# 检测设备类型
device_type = detect_device_type()
#向量索引路径
index_path = "./index" 


# 加载量化DDL描述模型
def load_ddl_vector_model(device_type):
    """
    根据设备类型加载适合的DDL向量化模型。
    :param device_type: 设备类型 ('gpu', 'cpu', 'apple_silicon')。
    :return: 加载的模型实例。
    """
    if device_type == "gpu":
        print("加载GPU优化的DDL模型 paraphrase-mpnet-base-v2...")
        return SentenceTransformer('paraphrase-mpnet-base-v2', device='cuda')  # 更强大但资源需求更高的模型
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

        # 向量化每一个 ddl_text
        vector = vectorize_ddl(ddl_text)
        
        # 保存向量到 FAISS 并获取 ID
        vector_ids = save_vector(vector)
        
        # 将 NumPy 数组转换为 Python 列表
        vector_list = vector.tolist()
        vector_ids_list = vector_ids.tolist()
        
        # 添加到结果列表
        all_vectors.append(vector_list)
        all_vector_ids.append(vector_ids_list)
        # 返回的格式类似于 {"vector_ids":[[0],[1]],"vector":[[0.07723584026098252,-0.06650960445404053,-0.07342207431793213],[0.07723584026098252,-0.06650960445404054,-0.07342207431793216]]}
        # 返回的是二维数组
    return {"vector_ids":all_vector_ids}

def save_vector(vectors: np.ndarray):
    defog = Defog()
    db_creds = defog.db_creds
    database_name = db_creds['database']  # 数据库名 

    faiss_manager = FaissManager(base_dir=index_path, dim=768)

    # 打印向量维度和类型以进行调试
    print("Adding vectors with shape:", vectors.shape)
    print("Data type:", vectors.dtype)
    
    print("==============~~~~~~~~~vectors shape:", vectors.shape)
    # 假设 vectors 是一维数组 
    print("vectors shape before reshape:", vectors.shape)
    if len(vectors.shape) == 1:
        vectors = vectors.reshape(1, -1)  # 转换成二维形状，(1, 768)
        print("=====是一维数组======")
    else:
        print("=====是二维数组======")
    print("vectors shape after reshape:", vectors.shape)



    # 确保向量维度正确并转换为 float32 类型
    if vectors.shape[1] != 768:
        raise ValueError(f"向量维度不匹配！索引需要 768 维，实际为 {vectors.shape[1]} 维。")
    
    vectors = vectors.astype(np.float32)  # 确保是 float32 类型

    vector_ids = faiss_manager.add_vectors(database_name, vectors)
    print(f"添加向量后返回的ID:")
    print(vector_ids)
    return vector_ids


#向量化文本
@router.post("/vectorize_ddl_test_save")
async def vectorize_ddl_test_save(request: Request):
    params = await request.json()
     # 获取传入的 ddl_text 数组
    ddl_texts = params.get("ddl_texts")
    return vectorize_ddl_save_ids(ddl_texts)







#向量化文本
@router.post("/vectorize_ddl_test_search")
async def vectorize_ddl_test_search(request: Request):
    params = await request.json()
  
    # 获取传入的 ddl_text 数组
    ddl_text = params.get("ddl_text")
    if not ddl_text:
        return {"error": "No ddl_text provided"}
    
    # 向量化一个 ddl_text
    vector = vectorize_ddl(ddl_text)
    
    defog = Defog()
    db_creds = defog.db_creds
    database_name = db_creds['database']  # 数据库名 
    top_k = 5  # 返回前 5 个最近邻

    faiss_manager = FaissManager(base_dir=index_path, dim=768)
    # 调用 search 方法进行查找
    indices = faiss_manager.search(database_name, vector, top_k)
    # 输出返回的索引（ID）
    print(f"查询到的最近邻索引ID: {indices}")
    #indices返回的类似于这样的二维数组，[[ 0  1 -1 -1 -1]]

    return indices



@router.post("/vectorize_delete_index")
async def vectorize_delete_index():
    defog = Defog()
    db_creds = defog.db_creds
    database_name = db_creds['database']  # 数据库名 

    faiss_manager = FaissManager(base_dir=index_path, dim=768)
    faiss_manager.delete_index(database_name)
    print(f"msg:", f"索引{database_name}:已删除!")
    return {"msg": f"索引{database_name}:已删除!"}