from fastapi import APIRouter, Request
import json
import os
from defog import Defog
from sqlcoder.vector_utils import (
    load_ddl_vector_model,
    vectorize_ddl,
    vectorize_ddl_save_ids,
    save_vector,
    search_vectorize,
    search_vectorize_join_ddl,
)
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



DEFOG_API_KEY = "NULL_VALUE" # placeholder, doesn't matter for any of the function here

home_dir = os.path.expanduser("~")
defog_path = os.path.join(home_dir, ".defog")

router = APIRouter()


    
@router.post("/integration/get_tables_db_creds")
async def get_tables_db_creds(request: Request):
    try:
        defog = Defog()
    except:
        return {"error": "no defog instance found"}

    try:
        with open(os.path.join(defog_path, "tables.json"), "r") as f:
            table_names = json.load(f)
    except:
        table_names = []

    try:
        with open(os.path.join(defog_path, "selected_tables.json"), "r") as f:
            selected_table_names = json.load(f)
    except:
        selected_table_names = []

    db_type = defog.db_type
    db_creds = defog.db_creds
    
    return {
        "tables": table_names,
        "db_creds": db_creds,
        "db_type": db_type,
        "selected_tables": selected_table_names
    }

@router.post("/integration/get_metadata")
async def get_metadata(request: Request):
    try:
        with open(os.path.join(defog_path, "metadata.json"), "r") as f:
            table_metadata = json.load(f)
        
        metadata = convert_nested_dict_to_list(table_metadata)
        return {"metadata": metadata}
    except:
        return {"error": "no metadata found"}
    
@router.post("/integration/get_metadata_json")
async def get_metadata(request: Request):
    try:
        with open(os.path.join(defog_path, "metadata.json"), "r") as f:
            table_metadata = json.load(f)
        
        # metadata = convert_nested_dict_to_list(table_metadata)
        metadata = table_metadata;
        return {"metadata": metadata}
    except:
        return {"error": "no metadata found"}
    

@router.post("/integration/generate_tables")
async def generate_tables(request: Request):
    params = await request.json()
    db_type = params.get("db_type")
    db_creds = params.get("db_creds")
    for k in ["api_key", "db_type"]:
        if k in db_creds:
            del db_creds[k]

    defog = Defog(DEFOG_API_KEY, db_type, db_creds)
    table_names = defog.generate_db_schema(tables=[], return_tables_only=True)

    with open(os.path.join(defog_path, "tables.json"), "w") as f:
        json.dump(table_names, f)

    return {"tables": table_names}

@router.post("/integration/generate_cloumn_metadata")
async def generate_cloumn_metadata(request: Request):
    params = await request.json()
    tables = params.get("tables")
    return generate_cloumn_metadata_json(tables=tables)


@router.post("/integration/generate_table_metadata")
async def generate_table_metadata(request: Request):
    params = await request.json()
    tables = params.get("tables")
    return generate_table_metadata_json(tables=tables)







@router.post("/integration/transform_column_metadata_to_ddl")
async def transform_column_metadata_to_ddl(request: Request):

    params = await request.json()
    tables = params.get("tables")
    # 返回最终的结果
    return get_column_to_ddl(tables=tables)


@router.post("/integration/transform_table_metadata_to_ddl")
async def transform_table_metadata_to_ddl(request: Request):

    params = await request.json()
    tables = params.get("tables")
    # 返回最终的结果
    return get_table_to_ddl(tables=tables)



@router.post("/integration/transform_column_metadata_to_vector_mix")
async def transform_column_metadata_to_vector_mix(request: Request):
    # 解析请求参数
    params = await request.json()
    tables = params.get("tables")

    # Step 1: 获取 DDL 文本
    ddl_response = await transform_column_metadata_to_ddl(request)  # 加上 await 等待异步结果
    # ddl_result = await ddl_response.json()  # 异步获取返回结果
    ddl_texts = ddl_response.get("ddl_texts")

    # Step 2: 向量化并获取向量 IDs
    if not ddl_texts:
        return {"error": "Failed to generate DDL texts"}
    
    vector_result = vectorize_ddl_save_ids(ddl_texts)  # 直接调用核心函数
    vector_ids = vector_result.get("vector_ids")

    # Step 3: 合并向量 IDs 和 DDL 文本
    if not vector_ids:
        return {"error": "Failed to process vectors"}

    mixed_ddl_texts = []
    for vector_id_list, ddl_text in zip(vector_ids, ddl_texts):
        if vector_id_list:  # 每个向量可能有多个 ID，这里取第一个
            vector_id = vector_id_list[0]
            mixed_ddl_text = f"vector_ids:{vector_id},{ddl_text}"
            mixed_ddl_texts.append(mixed_ddl_text)
    
   

    return {"ddl_texts": mixed_ddl_texts}



@router.post("/integration/transform_table_metadata_to_vector_mix")
async def transform_table_metadata_to_vector_mix(request: Request):
    # 解析请求参数
    params = await request.json()
    tables = params.get("tables")

    # Step 1: 获取 DDL 文本
    ddl_response = await transform_table_metadata_to_ddl(request)  # 加上 await 等待异步结果
    # ddl_result = await ddl_response.json()  # 异步获取返回结果
    ddl_texts = ddl_response.get("ddl_texts")

    # Step 2: 向量化并获取向量 IDs
    if not ddl_texts:
        return {"error": "Failed to generate DDL texts"}
    
    vector_result = vectorize_ddl_save_ids(ddl_texts)  # 直接调用核心函数
    vector_ids = vector_result.get("vector_ids")

    # Step 3: 合并向量 IDs 和 DDL 文本
    if not vector_ids:
        return {"error": "Failed to process vectors"}

    mixed_ddl_texts = []
    for vector_id_list, ddl_text in zip(vector_ids, ddl_texts):
        if vector_id_list:  # 每个向量可能有多个 ID，这里取第一个
            vector_id = vector_id_list[0]
            # 跳过空值或空字符串
            if not ddl_text.get('table_description') or ddl_text.get('table_description').strip() == "":
                continue
            #mixed_ddl_text = f"vector_ids:{vector_id},table_name:{ddl_text.get('table_name')},table_description:{ddl_text.get('table_description')}"
            mixed_ddl_texts.append( { "vector_ids":vector_id, "table_name":ddl_text.get('table_name'),"table_description":ddl_text.get('table_description')})

     # 返回合并后的结果
    metadata ={"ddl_texts": mixed_ddl_texts}

    # 写入JSON元数据
    with open(os.path.join(defog_path, "metadata_vector.json"), "w") as f:
        json.dump(metadata, f)
    print("json元数据文件metadata_vector.json写入完毕！")

    return metadata




@router.post("/integration/update_metadata")
async def update_metadata(request: Request):
    params = await request.json()
    metadata = params.get("metadata")

    # convert metadata to nested dictionary
    table_metadata = {}
    for item in metadata:
        table_name = item["table_name"]
        if table_name not in table_metadata:
            table_metadata[table_name] = []
        table_metadata[table_name].append(
            {
                "column_name": item["column_name"],
                "data_type": item["data_type"],
                "column_description": item["column_description"],
            }
        )
    
    with open(os.path.join(defog_path, "metadata.json"), "w") as f:
        json.dump(table_metadata, f)
    
    return {"status": "success"}

