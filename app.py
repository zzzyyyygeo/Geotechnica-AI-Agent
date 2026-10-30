# ==========================================
# 🚨 极其重要：环境变量必须在所有 import 之前设置！
# ==========================================
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
# 强制把工作目录切换到当前 app.py 所在的文件夹，防止路径找错
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import pdfplumber
import jieba
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from openai import OpenAI

# ==========================================
# 1. 智能查找 PDF 文件
# ==========================================
pdf_name = "地下工程防水技术规范.pdf"
possible_paths = [
    pdf_name,  
    os.path.join("data", "raw", pdf_name) 
]

pdf_path = None
for path in possible_paths:
    if os.path.exists(path):
        pdf_path = path
        break

if not pdf_path:
    print(f"\n❌ 找不到文件：{pdf_name}")
    print("请把 PDF 放到 1010 根目录下，或者 data/raw/ 文件夹下。")
    sys.exit()

print(f"✅ 成功定位 PDF 文件：{pdf_path}")
print("正在解析全书（使用 layout=True 保留表格结构），请稍等...")

# ==========================================
# 2. 解析整本 PDF（使用 layout 模式）
# ==========================================
docs = []
try:
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # 使用 layout=True 保留文字在页面上的位置，这样表格排版就不会全乱
            text = page.extract_text(layout=True) 
            if text and text.strip():
                docs.append(Document(page_content=text, metadata={"page": i+1}))
                
    # 过滤掉空块，防止向量数据库报错
    docs = [doc for doc in docs if doc.page_content.strip() != ""]
    print(f"✅ 全书解析完成！共提取了 {len(docs)} 个页面文档。")
except Exception as e:
    print(f"解析 PDF 发生错误: {e}")
    sys.exit()

# ==========================================
# 3. 构建混合检索（向量 + BM25）
# ==========================================
print("正在构建检索库（向量+BM25），请稍候...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma.from_documents(docs, embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

def preprocess_func(text):
    return list(jieba.cut(text))
bm25_retriever = BM25Retriever.from_documents(docs, preprocess_func=preprocess_func)
bm25_retriever.k = 10

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever], 
    weights=[0.5, 0.5]
)

# ==========================================
# 4. 加载 Rerank 精排模型
# ==========================================
print("正在加载 Rerank 重排模型（已开启离线模式，秒加载）...")
reranker = CrossEncoder("BAAI/bge-reranker-base")

# ==========================================
# 5. 配置大模型客户端
# ==========================================
# ⚠️⚠️⚠️ 务必替换成你自己的真实 API-KEY ⚠️⚠️⚠️
API_KEY = "sk-你的API-KEY" 
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com/v1")

# ==========================================
# 6. 聊天核心逻辑（融入常识兜底 Prompt）
# ==========================================
def chat_with_rag(message, history):
    # 1. 混合检索粗排
    candidates = ensemble_retriever.invoke(message)
    
    # 2. Rerank 精排
    pairs = [[message, doc.page_content] for doc in candidates]
    scores = reranker.predict(pairs)
    
    # 3. 排序取 Top 3
    scored_docs = list(zip(candidates, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in scored_docs[:3]]
    
    # 4. 拼接上下文
    context = "\n\n".join([doc.page_content for doc in top_docs])
    
    # 核心 Prompt：优先资料，没有资料则常识兜底
    prompt = f"""你是一个严谨的地下工程专家。
请优先根据以下参考资料回答用户的问题。
如果参考资料中完全没有答案，请直接说“资料中未找到”，然后**结合你自己的工程常识**给出专业建议，并明确标注“以下为AI补充知识”。

参考资料：
{context}

问题：{message}
"""
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"调用大模型失败，请检查 API Key 或网络：{e}"

# ==========================================
# 7. 搭建 Gradio 网页
# ==========================================
demo = gr.ChatInterface(
    fn=chat_with_rag,
    title="🏗️ 企业级地下工程规范知识库 (全能兜底版)",
    description="基于 layout 结构提取、混合检索与 Rerank。试试问我：'地下工程防水等级分为几级？各等级标准是什么？'"
)

if __name__ == "__main__":
    print("\n🚀 网页启动成功！请在浏览器中打开链接 (http://127.0.0.1:7860)")
    demo.launch()