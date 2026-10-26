from openai import OpenAI

client = OpenAI(
    api_key="sk-4c394acbd22f40ecb08edfb36e5bfc82",  # 替换成你的真实Key
    base_url="https://api.deepseek.com/v1"
)

# 用一个列表来存储对话历史（这就是AI的“记忆”）
messages = [
    {"role": "system", "content": "你是一个地下工程领域的专家，回答要专业、简洁。"}
]

print("开始对话（输入 'quit' 退出）")

while True:
    user_input = input("\n你：")
    if user_input == "quit":
        break
    
    # 1. 把用户的话加入历史
    messages.append({"role": "user", "content": user_input})
    
    # 2. 调用大模型
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )
    
    # 3. 获取AI的回答并加入历史
    ai_reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": ai_reply})
    
    print(f"AI：{ai_reply}")