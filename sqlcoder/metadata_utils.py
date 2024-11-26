import json
import os

from sqlcoder.kpaas_generate_schema  import KpaasGenerateSchema




home_dir = os.path.expanduser("~")
defog_path = os.path.join(home_dir, ".defog")



def detect_device_type():
    """
    检测设备类型，返回 'gpu', 'cpu', 或 'apple_silicon'。
    """
    import os
    import sys

    if os.popen("lspci | grep -i nvidia").read():
        return "gpu"
    elif sys.platform == "darwin" and os.uname().machine == "arm64":
        return "apple_silicon"
    else:
        return "cpu"




def convert_nested_dict_to_list(table_metadata):
    metadata = []
    for key in table_metadata:
        table_name = key
        for item in table_metadata[key]:
            item["table_name"] = table_name
            # if "column_description" not in item:
            #     item["column_description"] = ""
            metadata.append(item)
    return metadata

# 返回元数据JSON，参数是 表列表
def get_metadata_json(tables):
    
    # with open(os.path.join(defog_path, "selected_tables.json"), "w") as f:
    #     json.dump(tables, f)

    # defog = Defog()
    # metadata = defog.generate_db_schema(
    #     tables=tables, upload=False
    # )
    kpaas = KpaasGenerateSchema()
    metadata = kpaas.generate_cloumn_mysql_schema(
        tables=tables, upload=False
    )
    return metadata
    # print(f"执行了generate_mysql_schema_dev")
    # with open(os.path.join(defog_path, "metadata.json"), "w") as f:
    #     json.dump(metadata, f)
    
    # metadata = convert_nested_dict_to_list(metadata)
    # return {"metadata": metadata}


def generate_cloumn_metadata_json(tables):
    
    with open(os.path.join(defog_path, "selected_tables.json"), "w") as f:
        json.dump(tables, f)

    # defog = Defog()
    # metadata = defog.generate_db_schema(
    #     tables=tables, upload=False
    # )
    kpaas = KpaasGenerateSchema()
    metadata = kpaas.generate_cloumn_mysql_schema(
        tables=tables, upload=False
    )
    print(f"执行了generate_mysql_schema_dev")
    with open(os.path.join(defog_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    
    metadata = convert_nested_dict_to_list(metadata)
    return {"metadata": metadata}



def generate_table_metadata_json(tables):
    
    with open(os.path.join(defog_path, "selected_tables.json"), "w") as f:
        json.dump(tables, f)

    # defog = Defog()
    # metadata = defog.generate_db_schema(
    #     tables=tables, upload=False
    # )
    kpaas = KpaasGenerateSchema()
    metadata = kpaas.generate_table_mysql_schema(
        tables=tables, upload=False
    )
    print(f"执行了generate_mysql_schema_dev")
    with open(os.path.join(defog_path, "metadata_table.json"), "w") as f:
        json.dump(metadata, f)
    
    metadata = convert_nested_dict_to_list(metadata)
    return {"metadata": metadata}



def get_column_to_ddl(tables):
    metadata_json = generate_cloumn_metadata_json(tables=tables)
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

def get_table_to_ddl(tables):
    metadata_json = generate_table_metadata_json(tables=tables)
    ddl_texts = []

    print("========================metadata_json========================")
    print(metadata_json)
    

    # 遍历每一项元数据并构造相应的DDL文本
    for item in metadata_json.get("metadata", []):
        # 获取每一列的相关信息
        table_name = item.get("table_name", "")
        # column_name = item.get("column_name", "")
        # data_type = item.get("data_type", "")
        # column_description = item.get("column_description", "")
        table_description = item.get("table_description", "")
        
        if not table_description or table_description.strip() == "":
            continue

        # 创建对应的DDL文本格式
        # ddl_text = f"table_name: {table_name}, table_description: {table_description}"
        #ddl_text = table_description
        ddl_texts.append({ "table_name": table_name, "table_description": table_description })

    print("========================ddl_texts========================")
    print({"ddl_texts": ddl_texts})
    # 返回最终的结果
    return {"ddl_texts": ddl_texts}




def convert_metadata_to_ddl(metadata):
    # metadata is a dictionary of a table
    master_ddl = ""
    for table_name, columns in metadata.items():
        ddl = f"CREATE TABLE {table_name} (\n"
        table_description=""
        for column in columns:
            ddl += f"    {column['column_name']} {column['data_type']} COMMENT '{column['column_description']}',\n"
            table_description = column['table_description']
        ddl = ddl[:-2] + f"\n) COMMENT='{table_description}';"
        master_ddl += ddl + "\n\n"
    return master_ddl