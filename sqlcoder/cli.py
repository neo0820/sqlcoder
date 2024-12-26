import os
import sys
import sqlcoder
import subprocess
from huggingface_hub import snapshot_download, hf_hub_download
import asyncio

from sqlcoder.env_utils import server_ip, server_port

USAGE_STRING = """
Usage: sqlcoder <command>

Available commands:
    sqlcoder launch
    sqlcoder serve-webserver
    sqlcoder serve-static
"""

home_dir = os.path.expanduser("~")

def main():
    if len(sys.argv) < 2:
        print(USAGE_STRING)
        sys.exit(1)
    if sys.argv[1] == "launch":
        asyncio.run(launch())  # 在这里使用 asyncio.run 来运行异步函数
    elif sys.argv[1] == "serve-webserver":
        serve_webserver()
    elif sys.argv[1] == "serve-static":
        serve_static()
    else:
        print(USAGE_STRING)
        sys.exit(1)

def serve_webserver():
    from sqlcoder.serve import app
    from pyngrok import ngrok
    import uvicorn

    # port = 1235
    # 创建 ngrok 隧道
    # ngrok.set_auth_token("2ojSNGWUsSo7RKxo2Q5l6e1WVIX_4gCBB7sgrQBjpKVmDZp9t")  # 在这里替换为您的 ngrok 身份令牌
    # public_url = ngrok.connect(port)
    # print(f"Ngrok Tunnel URL: {public_url}")

    uvicorn.run(app, host=server_ip, port=server_port)


def serve_static():
    import http.server
    import socketserver
    import webbrowser

    port = 8002
    directory = os.path.join(sqlcoder.__path__[0], "static")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args,
                directory=directory,
                **kwargs
            )

    webbrowser.open(f"http://localhost:{port}")
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Static folder is {directory}")
        httpd.extension_maps = {".html": "text/html", "": "text/html"}
        httpd.serve_forever()

async def start_serve_static():
    static_process = subprocess.Popen(["sqlcoder", "serve-static"])
    print("Static server started.")
    await asyncio.sleep(1)  # 或者根据实际情况调整
    return static_process

async def start_serve_webserver():
    webserver_process = subprocess.Popen(["sqlcoder", "serve-webserver"])
    print("Webserver started.")
    await asyncio.sleep(1)
    return webserver_process


async def launch():
    home_dir = os.path.expanduser("~")
    defog_path = os.path.join(home_dir, ".defog")
    if not os.popen("lspci | grep -i nvidia").read():
        # not a GPU machine
        filepath = os.path.join(home_dir, ".defog", "llama-3-sqlcoder-8b.Q8_0.gguf")
        print("====执行路径====")
        print(filepath)
        if not os.path.exists(filepath):
            print(
                "Downloading the llama-3-sqlcoder-8b file. This is a ~5GB file and may take a long time to download. But once it's downloaded, it will be saved on your machine and you won't have to download it again."
            )
            hf_hub_download(repo_id="QuantFactory/llama-3-sqlcoder-8b-GGUF", filename="llama-3-sqlcoder-8b.Q8_0.gguf", local_dir=defog_path)
    else:
        # check if the huggingface model is already downloaded from hub. If not, download it
        from huggingface_hub import snapshot_download
        print(
            "Downloading the llama-3-sqlcoder-8b model. This is a ~14GB file and may take a long time to download. But once it's downloaded, it will be saved on your machine and you won't have to download it again."
        )
        _ = snapshot_download("defog/llama-3-sqlcoder-8b")
    
    # print("Starting SQLCoder server...")
    # static_process = subprocess.Popen(["sqlcoder", "serve-static"])
    #
    # print("Serving static server...")
    # webserver_process = subprocess.Popen(["sqlcoder", "serve-webserver"])

    # 启动静态和 Web 服务器的异步任务
    static_process = await start_serve_static()
    webserver_process = await start_serve_webserver()

    print("Press Ctrl+C to exit.")
    try:
        while True:
            await asyncio.sleep(1)  # 主线程保持活跃
    except KeyboardInterrupt:
        print("Exiting...")
        static_process.terminate()
        webserver_process.terminate()
        sys.exit(0)

if __name__ == "__main__":
    # main()
    asyncio.run(main())