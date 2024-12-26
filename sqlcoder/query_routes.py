from fastapi import APIRouter, Request
import os
import sys
import json
import sqlglot
import numpy as np


from defog import Defog
from defog.query import execute_query_once
from huggingface_hub import hf_hub_download

from sqlcoder.metadata_utils import (
    detect_device_type,
    convert_metadata_to_ddl,
    generate_cloumn_metadata_json,
    generate_table_metadata_json,
    get_column_to_ddl,
    get_metadata_json,
    get_table_to_ddl,
)

from sqlcoder.query_utils import (
    get_query_by_nl_test,
    get_query_by_nl,
    get_query_by_nl_step1,
    get_query_by_nl_step2,
    get_query_by_nl_step3,
    get_query_by_nl_step4,
    get_device_type,
    convert_sql,
    load_sql_model,
    generate_function, send_message_to_ollama, chat_ollama
)

router = APIRouter()

home_dir = os.path.expanduser("~")
defog_path = os.path.join(home_dir, ".defog")

generate_function = load_sql_model()

@router.post("/get_device_type")
async def get_device_type():
    return detect_device_type()

@router.post("/query_test")
async def query_test(request: Request):
    body = await request.json()
    question = body.get("question")
    return get_query_by_nl_test(question)

@router.post("/query")
async def query(request: Request):

    params = await request.json()
    # 获取传入的 question 问题
    question = params.get("question")
    # 设置最近邻数量
    top_k  = params.get("top_k")
    # 距离阈值，用于筛选相关结果
    distance_threshold = params.get("distance_threshold")

    return get_query_by_nl(question,top_k,distance_threshold)    

@router.post("/query_step1")
async def query_step1(request: Request):
    params = await request.json()
    # 获取传入的 question 问题
    question = params.get("question")
    # 设置最近邻数量
    top_k  = params.get("top_k")
    # 距离阈值，用于筛选相关结果
    distance_threshold = params.get("distance_threshold")

    return get_query_by_nl_step1(question,top_k,distance_threshold)

import asyncio
@router.post("/query_step2")
async def query_step2(request: Request):
    params = await request.json()
    return await get_query_by_nl_step2(params)

@router.post("/query_step3")
async def query_step3(request: Request):
    params = await request.json()
    return await get_query_by_nl_step3(params)

@router.post("/query_step4")
async def query_step4(request: Request):
    params = await request.json()
    return get_query_by_nl_step4(params)

@router.post("/query_step5")
async def query_step5(request: Request):
    params = await request.json()
    return send_message_to_ollama("今天天气怎么样")

