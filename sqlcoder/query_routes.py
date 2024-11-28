from fastapi import APIRouter, Request
import os
import sys
import json
import sqlglot
import numpy as np
import torch

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
    query,
    get_device_type,
    convert_sql,
    load_sql_model,
)

router = APIRouter()

home_dir = os.path.expanduser("~")
defog_path = os.path.join(home_dir, ".defog")
generate_function = load_sql_model()

@router.post("/get_device_type")
async def get_device_type():
    return detect_device_type()

@router.post("/query")
async def query(request: Request):
    body = await request.json()
    question = body.get("question")
    
    torch.cuda.empty_cache()
    print("==================1111111")
    with open(os.path.join(defog_path, "metadata.json"), "r") as f:
        metadata = json.load(f)
    print("==================2222222")
    ddl = convert_metadata_to_ddl(metadata)
    print("==================3333333")
    prompt = f"""### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Instructions
- If you cannot answer the question with the available database schema, return 'I do not know'

### Database Schema
The query will run on a database with the following schema:
{ddl}

### Answer
Given the database schema, here is the SQL query that answers [QUESTION]{question}[/QUESTION]
[SQL]
"""
    print("==================prompt")
    print(prompt)
    
    query = generate_function(prompt)
    print("==================44444444")
    defog = Defog()
    print(f"defog.db_type: {defog.db_type}")
    
    db_type = defog.db_type or "postgres"
    db_creds = defog.db_creds
    query = convert_sql(query,source_db="postgres", target_db=db_type)
    
    columns, data = execute_query_once(db_type, db_creds, query)
    
    return {
        "ddl": ddl,
        "prompt": prompt,
        "query_generated": query,
        "columns": columns,
        "data": data,
        "ran_successfully": True,
        "db_type": db_type
    }
    



