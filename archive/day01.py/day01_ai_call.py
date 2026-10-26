from openai import OpenAI

# 初始化客户端（注意：DeepSeek的API地址和OpenAI不同）
client = OpenAI(
    api_key="sk-4c394acbd22f40ecb08edfb36e5bfc82",  # 替换成你的真实Key
    base_url="https://api.deepseek.com/v1"
)

# 设置提问内容（结合你的专业）
messages = [
    {"role": "system", "content": "你是一个地下工程领域的专家，请用简短易懂的话回答。"},
    {"role": "user", "content": "什么是盾构法施工？它有什么优缺点？"}
]

# 调用大模型
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages
)

# 打印结果
print("AI 回答：")
print(response.choices[0].message.content)