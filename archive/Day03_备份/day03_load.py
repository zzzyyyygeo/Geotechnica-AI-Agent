from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import os

# 路径对应你左侧的文件夹结构
loaders = {
    "data/地下工程防水技术规范.pdf": PyPDFLoader,
    "data/report.docx": Docx2txtLoader,
    "data/notes.txt": lambda path: TextLoader(path, encoding="utf-8")
}

all_documents = []

for file_path, loader_class in loaders.items():
    if os.path.exists(file_path):
        print(f"正在加载: {file_path}")
        loader = loader_class(file_path)
        docs = loader.load()
        all_documents.extend(docs)
        print(f"  - 成功加载 {len(docs)} 页/段落")
    else:
        print(f"⚠️ 文件不存在: {file_path}")

print(f"\n总共加载了 {len(all_documents)} 个文档单元")
print("=" * 50)

if all_documents:
    print("第一个文档单元的元数据：")
    print(all_documents[0].metadata)
    print("\n第一段内容预览：")
    print(all_documents[0].page_content[:200])
