import os
import sys
import json
import sqlglot
import numpy as np

from defog import Defog
from defog.query import execute_query_once
from huggingface_hub import hf_hub_download

from sqlcoder.env_utils import proxy_ip, proxy_port, model_sql_handler, model_qwen25_handler
from sqlcoder.ollama_utils import generate_response_url, chat_with_model_url, call_ollama_generate_response_stream, \
    call_ollama_generate_response
from sqlcoder.vector_utils import (
    search_vectorize,
    search_vectorize_2_table,
    search_table_2_ddl,
    vectorize_ddl,
    vectorize_ddl_save_ids,
    load_ddl_vector_model,
    delete_index,
)

from sqlcoder.metadata_utils import (
    detect_device_type,
    convert_metadata_to_ddl,
    generate_table_metadata_json,
    generate_cloumn_metadata_json,
    get_metadata_json,
    convert_nested_dict_to_list,
    get_column_to_ddl,
    get_table_to_ddl,
)

home_dir = os.path.expanduser("~")
defog_path = os.path.join(home_dir, ".defog")

device_type = detect_device_type()


def get_device_type():
    return detect_device_type()


# 加载SQL生成模型
def load_sql_model():
    if device_type == "gpu":
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        model = AutoModelForCausalLM.from_pretrained(
            "defog/llama-3-sqlcoder-8b",
            device_map="auto",
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained("defog/llama-3-sqlcoder-8b")
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
        return lambda prompt: pipe(
            prompt,
            max_new_tokens=512,
            do_sample=False,
            num_beams=3,
            num_return_sequences=1,
            return_full_text=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )[0]["generated_text"].split(";")[0].split("```")[0].strip() + ";"
    else:
        from llama_cpp import Llama

        filepath = os.path.join(defog_path, "llama-3-sqlcoder-8b.Q8_0.gguf")

        if not os.path.exists(filepath):
            print(
                "Downloading the llama-3-sqlcoder-8b  GGUF file. This is a 4GB file and may take a long time to download. But once it's downloaded, it will be saved on your machine and you won't have to download it again."
            )

            # 下载 GGUF 文件
            hf_hub_download(repo_id="QuantFactory/llama-3-sqlcoder-8b-GGUF", filename="llama-3-sqlcoder-8b.Q8_0.gguf",
                            local_dir=defog_path)

        if device_type == "apple_silicon":
            llm = Llama(model_path=filepath, n_gpu_layers=-1, n_ctx=4096)
        else:
            llm = Llama(model_path=filepath, n_ctx=4096)

        return lambda prompt: llm(
            prompt,
            max_tokens=512,
            temperature=0,
            top_p=1,
            echo=False,
            repeat_penalty=1.0
        )["choices"][0]["text"].split(";")[0].split("```")[0].strip() + ";"


#注意，这里load_sql_model别让多次加载，执行一次，别的地方调用generate_function取值就好，否则会出现gpu内存溢出的问题
generate_function = load_sql_model()


def get_query_by_nl(question, top_k, distance_threshold):
    #分段执行
    vectorize_table_json = get_query_by_nl_step1(question, top_k, distance_threshold)

    return get_query_by_nl_step2(vectorize_table_json)


def get_query_by_nl_step1(question, top_k, distance_threshold):
    import torch
    torch.cuda.empty_cache()
    # with open(os.path.join(defog_path, "metadata.json"), "r") as f:
    #     metadata = json.load(f)
    # ddl = convert_metadata_to_ddl(metadata)

    vectorize_table_json = search_vectorize_2_table(question, top_k, distance_threshold)
    vectorize_table_json["question"] = question
    return vectorize_table_json

#
#
# def get_query_by_nl_step2(vectorize_table_json):
#     question = vectorize_table_json.get("question")
#     table_names = vectorize_table_json.get("table_names")
#     table_descriptions = vectorize_table_json.get("table_descriptions")
#
#     table_2_ddl_json = search_table_2_ddl(table_names)
#     ddl = table_2_ddl_json.get("ddl")
#
#     prompt = f"""### Task
# Generate a SQL query to answer [QUESTION]{question}[/QUESTION]
#
# ### Instructions
# - If you cannot answer the question with the available database schema, return 'I do not know'
#
# ### Database Schema
# The query will run on a database with the following schema:
# {ddl}
#
# ### Answer
# Given the database schema, here is the SQL query that answers [QUESTION]{question}[/QUESTION]
# [SQL]
# """
#     query = generate_function(prompt)
#     defog = Defog()
#     print(f"defog.db_type: {defog.db_type}")
#
#     db_type = defog.db_type or "postgres"
#     db_creds = defog.db_creds
#     query = convert_sql(query,source_db="postgres", target_db=db_type)
#
#     # columns, data = execute_query_once(db_type, db_creds, query)
#
#     return {
#
#         "table_descriptions": table_descriptions,
#         "table_names":  table_names,
#         #"ddl": ddl,
#         "prompt": prompt,
#         "sql": query,
#         # "columns": columns,
#         # "data": data,
#         #"ran_successfully": True,
#
#         "db_type": db_type
#     }



import requests


async def get_query_by_nl_step2(vectorize_table_json):
    question = vectorize_table_json.get("question")
    table_names = vectorize_table_json.get("table_names")
    table_descriptions = vectorize_table_json.get("table_descriptions")

    table_2_ddl_json = search_table_2_ddl(table_names)
    ddl = table_2_ddl_json.get("ddl")

    model = model_sql_handler

    prompt = f"""### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Instructions
- If you cannot answer the question with the available database schema, return 'I do not know'

### Database Schema
The query will run on a database with the following schema:
{ddl}

### Answer
Given the database schema, here is the SQL query that answers [QUESTION]{question}[/QUESTION]
Add aliases to the fields after SQL query, where the aliases are derived from the COMMENT in the Database Schema
[SQL]
"""
    header = {"Content-Type": "application/json"}

    response = await call_ollama_generate_response_stream(model, prompt, header)


    # 对返回的流文本进行合并处理。
    response_text = response.replace("\n", "").replace("\r", "").replace("}", "},")
    response_text = "["+response_text +"]"
    response_text = response_text.replace(",]", "]")
    # 将替换后的文本解析成JSON
    try:
        response_json = json.loads(response_text)
        # 提取并合并所有的 'response' 字段
        response_contents = [
            item["response"] for item in response_json if "response" in item
        ]
        # 合并所有的response内容成一个字符串
        merged_response = "".join(response_contents)
        # 返回合并后的响应
        return {"sql": merged_response, "table_names": table_names}
    except json.JSONDecodeError:
        print("Failed to decode JSON")
    except KeyError as e:
        print(f"KeyError: {e}")


async def get_query_by_nl_step3(vectorize_table_json):
    question = vectorize_table_json.get("question")
    table_names = vectorize_table_json.get("table_names")
    table_descriptions = vectorize_table_json.get("table_descriptions")

    table_2_ddl_json = search_table_2_ddl(table_names)
    ddl = table_2_ddl_json.get("ddl")

    model = model_sql_handler

    prompt = f"""### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Instructions
- If you cannot answer the question with the available database schema, return 'I do not know'

### Database Schema
The query will run on a database with the following schema:
{ddl}

### Answer
Given the database schema, here is the SQL query that answers [QUESTION]{question}[/QUESTION]
Add aliases to the fields after SQL query, where the aliases are derived from the COMMENT in the Database Schema
[SQL]
"""
    header = {"Content-Type": "application/json"}

    response = await call_ollama_generate_response(model, prompt, header)
    print(response)
    return {"sql": response, "table_names": table_names}


async def get_query_atfer_alias(query_json):
    table_names = query_json.get("table_names")
    table_2_ddl_json = search_table_2_ddl(table_names)
    ddl = table_2_ddl_json.get("ddl")

    sql = query_json.get("sql")

    model = model_qwen25_handler

    prompt = f"""表的DDL是这样子的
    {ddl}
    
    然后sql查询是这样子的
    {sql}
    
    把上面SQL查询的字段加上别名，别名的来源是DDL中的COMMENT，并显示出来，注意：以SQL中的字段为准，别多选多余的字段 [sql]
"""
    header = {"Content-Type": "application/json"}

    response = await call_ollama_generate_response(model, prompt, header)
    print(response)
    return {"sql": get_sql_ony(response)}


import re
def get_sql_ony(response):
    # 使用正则表达式匹配 SQL 语句
    sql_pattern = re.compile(r'```sql\n(.*?)\n```', re.DOTALL)

    # 搜索并提取 SQL 语句
    match = sql_pattern.search(response)
    if match:
        sql_query = match.group(1).strip()
        print("抽取出来的sql如下：")
        print(sql_query)
        return sql_query
    else:
        print("未找到SQL语句")


def get_query_by_nl_test(question):
    import torch
    torch.cuda.empty_cache()
    with open(os.path.join(defog_path, "metadata.json"), "r") as f:
        metadata = json.load(f)
    ddl = convert_metadata_to_ddl(metadata)
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
    query = generate_function(prompt)
    defog = Defog()
    print(f"defog.db_type: {defog.db_type}")

    db_type = defog.db_type or "postgres"
    db_creds = defog.db_creds
    query = convert_sql(query, source_db="postgres", target_db=db_type)
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


# 定义一个函数，用于将 SQL 查询从一种数据库类型转换为另一种
def convert_sql(query: str, source_db: str, target_db: str) -> str:
    try:
        # 使用 sqlglot 解析 SQL 语句并转换为目标数据库方言
        converted_query = sqlglot.transpile(query, read=source_db, write=target_db)[0]
        return converted_query
    except Exception as e:
        print(f"SQL conversion error: {e}")
        return None







def get_query_by_nl_step4(vectorize_table_json):
    question = vectorize_table_json.get("question")
    table_names = vectorize_table_json.get("table_names")
    table_descriptions = vectorize_table_json.get("table_descriptions")

    table_2_ddl_json = search_table_2_ddl(table_names)
    ddl = table_2_ddl_json.get("ddl")

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

    # Call Ollama API to get the SQL query
    url = generate_response_url
    payload = {
        "model": "mannix/defog-llama3-sqlcoder-8b:latest",
        "prompt": prompt
    }
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        #
        # return response.text
        # 获取response.text并替换换行符
        response_text = response.text.replace("\n", "").replace("\r", "").replace("}", "},")

        response_text = "["+response_text +"]"
        response_text = response_text.replace(",]", "]")
        # 将替换后的文本解析成JSON
        try:
            response_json = json.loads(response_text)

            # 提取并合并所有的 'response' 字段
            response_contents = [
                item["response"] for item in response_json if "response" in item
            ]

            # 合并所有的response内容成一个字符串
            merged_response = "".join(response_contents)

            # 返回合并后的响应
            return {"sql": merged_response}
        except json.JSONDecodeError:
            print("Failed to decode JSON")
        except KeyError as e:
            print(f"KeyError: {e}")
    else:
        return {"sql": response.text}


def send_message_to_ollama(message):
    print(f"chat_with_model_url:{chat_with_model_url}")
    url = chat_with_model_url
    payload = {
        "model": "Qwen2.5:7b",
        "messages": [{"role": "user", "content": message}]
    }
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        response_content = ""
        for line in response.iter_lines():
            if line:
                response_content += json.loads(line)["message"]["content"]
        return response_content
    else:
        return f"Error: {response.status_code} - {response.text}"


from ollama import chat
async def chat_ollama():
    stream = chat(
        model='Qwen2.5:7b',
        messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
        stream=True,
    )
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)