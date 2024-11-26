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
from sqlcoder.vector_utils import load_ddl_vector_model, vectorize_ddl, vectorize_ddl_save_ids, search_vectorize, search_vectorize_join_ddl


router = APIRouter()

#向量索引路径
index_path = "./index" 


#向量化文本
@router.post("/vectorize_ddl_test_save")
async def vectorize_ddl_test_save(request: Request):
    params = await request.json()
     # 获取传入的 ddl_text 数组
    ddl_texts = params.get("ddl_texts")
    return vectorize_ddl_save_ids(ddl_texts)


#向量化文本
@router.post("/search_vectorize_json")
async def search_vectorize_json(request: Request):
    params = await request.json()
  
    # 获取传入的 search_text 数组
    search_text = params.get("search_text")
    # 设置最近邻数量
    top_k  = params.get("top_k")
    # 距离阈值，用于筛选相关结果
    distance_threshold = params.get("distance_threshold")

    return search_vectorize(search_text, top_k, distance_threshold)

#向量化文本
@router.post("/search_vectorize_join_ddl_json")
async def search_vectorize_join_ddl_json(request: Request):
    params = await request.json()
  
    # 获取传入的 search_text 数组
    search_text = params.get("search_text")
    # 设置最近邻数量
    top_k  = params.get("top_k")
    # 距离阈值，用于筛选相关结果
    distance_threshold = params.get("distance_threshold")

    return search_vectorize_join_ddl(search_text, top_k, distance_threshold)



@router.post("/vectorize_delete_index")
async def vectorize_delete_index():
    defog = Defog()
    db_creds = defog.db_creds
    database_name = db_creds['database']  # 数据库名 

    faiss_manager = FaissManager(base_dir=index_path, dim=768)
    faiss_manager.delete_index(database_name)
    print(f"msg:", f"索引{database_name}:已删除!")
    return {"msg": f"索引{database_name}:已删除!"}