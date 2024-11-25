from fastapi import APIRouter, Request
import os
import sys
import json
import sqlglot
import numpy as np

from defog import Defog
from defog.query import execute_query_once
from huggingface_hub import hf_hub_download

from sentence_transformers import SentenceTransformer
from sqlcoder.faiss_manager import FaissManager


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

def load_ddl_vector_model(device_type):
    """
    根据设备类型加载适合的DDL向量化模型。
    """
    if device_type == "gpu":
        print("加载GPU优化的DDL模型 paraphrase-mpnet-base-v2...")
        return SentenceTransformer('paraphrase-mpnet-base-v2', device='cuda')
    elif device_type == "apple_silicon":
        print("加载适用于 Apple Silicon 的DDL模型 paraphrase-MiniLM-L6-v2...")
        return SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')
    else:
        print("加载轻量化的DDL模型 paraphrase-MiniLM-L6-v2 (CPU 模式)...")
        return SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')

# 检测设备类型
device_type = detect_device_type()

# 加载DDL向量化模型
ddl_vector_model = load_ddl_vector_model(device_type)

# 测试模型加载和向量化
ddl_text = "表名: base_dictionarydata, 列名: F_Id, 数据类型: varchar, 列描述: , 表描述: 字典数据"
vector = ddl_vector_model.encode(ddl_text)
print("DDL向量化结果:", vector)

router = APIRouter()

device_type = None
generate_function = None
ddl_vector_model = None  # 量化模型实例

index_path = "./index" #向量索引路径

DEFOG_API_KEY = "NULL_VALUE" # placeholder, doesn't matter for any of the function here

home_dir = os.path.expanduser("~")
defog_path = os.path.join(home_dir, ".defog")

# stuff that we need to do only once, before everything is loaded

# 检测设备类型
def detect_device_type():
    if os.popen("lspci | grep -i nvidia").read():
        return "gpu"
    elif sys.platform == "darwin" and os.uname().machine == "arm64":
        return "apple_silicon"
    else:
        return "cpu"

device_type = detect_device_type()


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

        filepath = os.path.join(defog_path, "sqlcoder-7b-q5_k_m.gguf")

        if not os.path.exists(filepath):
            print(
                "Downloading the SQLCoder-7b GGUF file. This is a 4GB file and may take a long time to download. But once it's downloaded, it will be saved on your machine and you won't have to download it again."
            )

            # 下载 GGUF 文件
            hf_hub_download(repo_id="defog/sqlcoder-7b-2", filename="sqlcoder-7b-q5_k_m.gguf", local_dir=defog_path)

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

generate_function = load_sql_model()


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

#向量化文本
@router.post("/vectorize_ddl_test_save")
async def vectorize_ddl_test_save(request: Request):
    params = await request.json()
  
     # 获取传入的 ddl_text 数组
    ddl_texts = params.get("ddl_texts")
    
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

    return {"vector_ids":all_vector_ids,"vector": all_vectors }


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

    return {"indices":indices.tolist()[0]}



@router.post("/vectorize_delete_index")
async def vectorize_delete_index():
    defog = Defog()
    db_creds = defog.db_creds
    database_name = db_creds['database']  # 数据库名 

    faiss_manager = FaissManager(base_dir=index_path, dim=768)
    faiss_manager.delete_index(database_name)
    print(f"msg:", f"索引{database_name}:已删除!")
    return {"msg": f"索引{database_name}:已删除!"}


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



@router.post("/get_device_type")
async def get_device_type():
    return {"device_type": device_type}

@router.post("/query")
async def query(request: Request):
    body = await request.json()
    question = body.get("question")
    
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


# 定义一个函数，用于将 SQL 查询从一种数据库类型转换为另一种
def convert_sql(query: str, source_db: str, target_db: str) -> str:
    try:
        # 使用 sqlglot 解析 SQL 语句并转换为目标数据库方言
        converted_query = sqlglot.transpile(query, read=source_db, write=target_db)[0]
        return converted_query
    except Exception as e:
        print(f"SQL conversion error: {e}")
        return None