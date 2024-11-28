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
    get_query_by_nl,
    get_device_type,
    convert_sql,
    load_sql_model,
    generate_function
)

router = APIRouter()

home_dir = os.path.expanduser("~")
defog_path = os.path.join(home_dir, ".defog")

# generate_function = load_sql_model()

@router.post("/get_device_type")
async def get_device_type():
    return detect_device_type()

@router.post("/query")
async def query(request: Request):
    body = await request.json()
    question = body.get("question")
    return get_query_by_nl(question)
    



