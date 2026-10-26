from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import os

# 打印当前目录，确认工作目录正确
print("当前工作目录:", os.getcwd())
print("目录下的真实文件:", os.listdir("."))

# 1. 加载文档（直接在当前目录下找）
loader = PyPDFLoader("地下工程防水技术规范.pdf")
pages = loader.load()

# 2. 创建切分器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "；", "！", "？", "，", "、", " ", ""]
)

# 3. 执行切分
chunks = text_splitter.split_documents(pages)
print(f"文档被切分成了 {len(chunks)} 个小块")

# 4. 把切分结果保存成 JSON 文件
output_data = []
for i, chunk in enumerate(chunks):
    output_data.append({
        "id": i,
        "content": chunk.page_content,
        "metadata": chunk.metadata
    })

with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print("✅ 切分结果已保存到 chunks.json")
