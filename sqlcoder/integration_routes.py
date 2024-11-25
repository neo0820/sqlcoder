from fastapi import APIRouter, Request
import json
import os
from defog import Defog
from sqlcoder.kpaas_generate_schema  import KpaasGenerateSchema
from sqlcoder.vector_routes import vectorize_ddl_save_ids, save_vector, load_ddl_vector_model, vectorize_ddl, vectorize_delete_index




DEFOG_API_KEY = "NULL_VALUE" # placeholder, doesn't matter for any of the function here

home_dir = os.path.expanduser("~")
defog_path = os.path.join(home_dir, ".defog")

router = APIRouter()

def convert_nested_dict_to_list(table_metadata):
    metadata = []
    for key in table_metadata:
        table_name = key
        for item in table_metadata[key]:
            item["table_name"] = table_name
            if "column_description" not in item:
                item["column_description"] = ""
            metadata.append(item)
    return metadata


    
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

@router.post("/integration/generate_metadata")
async def generate_metadata(request: Request):
    params = await request.json()
    tables = params.get("tables")
    return generate_metadata_json(tables=tables)

def generate_metadata_json(tables):
    
    with open(os.path.join(defog_path, "selected_tables.json"), "w") as f:
        json.dump(tables, f)

    # defog = Defog()
    # metadata = defog.generate_db_schema(
    #     tables=tables, upload=False
    # )
    kpaas = KpaasGenerateSchema()
    metadata = kpaas.generate_mysql_schema(
        tables=tables, upload=False
    )
    print(f"执行了generate_mysql_schema_dev")
    with open(os.path.join(defog_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    
    metadata = convert_nested_dict_to_list(metadata)
    return {"metadata": metadata}



@router.post("/integration/transform_metadata_to_ddl")
async def transform_metadata_to_ddl(request: Request):

    params = await request.json()
    tables = params.get("tables")
    # 返回最终的结果
    return get_table_to_ddl(tables=tables)




@router.post("/integration/transform_metadata_to_vector_mix")
async def transform_metadata_to_vector_mix(request: Request):
    # 解析请求参数
    params = await request.json()
    tables = params.get("tables")

    # Step 1: 获取 DDL 文本
    ddl_response = await transform_metadata_to_ddl(request)  # 加上 await 等待异步结果
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

    # 返回合并后的结果
    return {"ddl_texts": mixed_ddl_texts}




def get_table_to_ddl(tables):
    metadata_json = generate_metadata_json(tables=tables)
    ddl_texts = []

    print("========================metadata_json========================")
    print(metadata_json)
    

    # 遍历每一项元数据并构造相应的DDL文本
    for item in metadata_json.get("metadata", []):
        # 获取每一列的相关信息
        table_name = item.get("table_name", "")
        column_name = item.get("column_name", "")
        data_type = item.get("data_type", "")
        column_description = item.get("column_description", "")
        table_description = item.get("table_description", "")

        # 创建对应的DDL文本格式
        ddl_text = f"table_name: {table_name}, column_name: {column_name}, data_type: {data_type}, column_description: {column_description}, table_description: {table_description}"
        ddl_texts.append(ddl_text)

    print("========================ddl_texts========================")
    print({"ddl_texts": ddl_texts})
    # 返回最终的结果
    return {"ddl_texts": ddl_texts}


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

