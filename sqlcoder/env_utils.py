import os
from dotenv import load_dotenv

# 根据环境变量加载不同的配置文件
match os.getenv("ENVIRONMENT"):
    case "dev":
        # 开发环境：加载 .env.dev 文件
        load_dotenv(".env.dev")
    case "docker":
        # Docker环境：加载 .env.docker 文件
        load_dotenv(".env.docker")
    case _:
        # 默认是开发环境：加载 .env.dev 文件
        load_dotenv(".env.dev")

server_ip = os.getenv("SERVER_IP")
server_port = os.getenv("SERVER_PORT")

ollama_ip = os.getenv("OLLAMA_IP")
ollama_port = os.getenv("OLLAMA_PORT")

model_sql_handler = os.getenv("MODEL_SQL_HANDLER")
