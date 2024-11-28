from fastapi import APIRouter, Request
import os
import sys
import json

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
from sqlcoder.vector_utils import (
    vectorize_ddl,
    vectorize_ddl_save_ids,
    save_vector,
    search_vectorize,
    search_vectorize_2_table,
    load_ddl_vector_model,
    search_table_2_ddl,
    delete_index
)

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
  
    # 获取传入的 question 问题
    question = params.get("question")
    # 设置最近邻数量
    top_k  = params.get("top_k")
    # 距离阈值，用于筛选相关结果
    distance_threshold = params.get("distance_threshold")

    return search_vectorize(question, top_k, distance_threshold)

#查找文本，返回其向量相关的表结构
@router.post("/search_vectorize_2_table_json")
async def search_vectorize_2_table_json(request: Request):
    params = await request.json()
  
    # 获取传入的 question 问题
    question = params.get("question")
    # 设置最近邻数量
    top_k  = params.get("top_k")
    # 距离阈值，用于筛选相关结果
    distance_threshold = params.get("distance_threshold")

    return search_vectorize_2_table(question, top_k, distance_threshold)


#通过表名返回表的元数据。
@router.post("/search_table_2_ddl_json")
async def search_table_2_ddl_json(request: Request):
    params = await request.json()
    
    table_names = params.get("table_names")
    return search_table_2_ddl(table_names)


@router.post("/vectorize_delete_index")
async def vectorize_delete_index():
    return delete_index(index_path)