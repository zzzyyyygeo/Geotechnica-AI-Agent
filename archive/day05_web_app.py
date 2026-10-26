import os
# 👇 这三行必须放在最上面！设置国内镜像 + 强制离线
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"          # 强制HuggingFace不访问外网
os.environ["TRANSFORMERS_OFFLINE"] = "1"    # 强制Transformers库不访问外网

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI
import gradio as gr

# ... 下面的代码完全不用改 ...
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI
import gradio as gr

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("正在加载向量库...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
# 直接加载刚才建好的本地向量库，秒开！
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

client = OpenAI(
    api_key="你的API-KEY", # ⚠️ 填入最新的 Key
    base_url="https://api.deepseek.com/v1"
)

def chat_with_rag(message, history):
    # 1. 检索
    results = vectorstore.similarity_search(message, k=3)
    context = "\n".join([doc.page_content for doc in results])
    
    # 2. 构建 Prompt
    prompt = f"""你是一个地下工程领域的专家。请根据以下参考资料回答用户的问题。
如果参考资料中没有答案，请直接说“资料中未找到相关内容”，不要编造。

参考资料：
{context}

用户问题：{message}
"""
    # 3. 调用大模型
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 4. 搭建 Gradio 界面
demo = gr.ChatInterface(
    fn=chat_with_rag,
    title="🏗️ 地下工程规范智能助手",
    description="基于《地下工程防水技术规范》等 416 个碎块，支持自然语言问答。",
    
)

if __name__ == "__main__":
    demo.launch()