from sqlcoder.env_utils import proxy_ip, proxy_port, proxy_protocol

generate_response_url = f"{proxy_protocol}://{proxy_ip}:{proxy_port}/api/generate"
chat_with_model_url = f"{proxy_protocol}://{proxy_ip}:{proxy_port}/api/chat"

import aiohttp
import json

# generate_response 流式返回
# model 模型名称
# prompt 提示词
# header 头部信息
async def call_ollama_generate_response_stream(model, prompt, header):
    url = generate_response_url
    payload = {
        "model": model,
        "prompt": prompt
    }
    headers = header

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as response:
            # 逐步处理流式响应
            response_text = ""
            async for chunk in response.content.iter_any():
                response_text += chunk.decode('utf-8')
                print("Partial response:", response_text)  # 打印每部分响应

                # 当完整响应到达时，处理结果
                if '"done": true' in response_text:
                    print("Final response received")
                    break

            return response_text


async def call_ollama_generate_response(model, prompt, header):
    url = generate_response_url
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    headers = header

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as response:

            # 直接获取完整响应内容
            response_text = await response.text()
            # 打印完整的响应
            #print("Final response:", response_text)

            # 解析JSON响应
            response_json = json.loads(response_text)

            # 提取response属性
            response_data = response_json.get('response', None)

            # 打印response数据
            #print("Response:", response_data)

            return response_data
