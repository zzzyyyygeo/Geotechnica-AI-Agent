# 101# 🏗️ 地下工程智能规范 Agent 助手 (Geotechnical RAG Agent)

## 📖 项目简介
本项目是一个专为地下工程与智能建造领域打造的 AI 智能体（Agent）。它不仅能基于国家/行业规范（如《地下工程防水技术规范》）进行精准的**知识检索问答（RAG）**，还能通过 **Agent 智能调度**，自主调用数学工具进行**基坑土方量计算**，极大降低了通用大模型在垂直工程领域的“幻觉”问题。

## ✨ 核心功能
1. **防幻觉问答 (RAG)**：基于 178 页真实 PDF 规范，切分为 400+ 碎块，生成并持久化本地向量库。回答严格附带规范来源，拒绝 AI 胡编乱造。
2. **智能体调度 (Agent)**：大模型可自主判断用户意图。遇到规范问题调用检索工具（RAG），遇到算量问题调用计算工具（Calculator），支持复合型多任务处理。
3. **工程化结构**：采用标准目录结构（`src/`、`data/`、`app.py` 分离），并隔离敏感配置（`.env`），具备极高的可维护性。
4. **Web 交互界面**：基于 Gradio 构建，支持自然语言实时对话。

## 🛠️ 技术栈
*   **大模型 & 编排**：LangChain (Agent, RAG), DeepSeek API
*   **向量数据库**：ChromaDB
*   **Embedding 模型**：BAAI/bge-small-zh-v1.5
*   **Web 框架**：Gradio
*   **核心语言**：Python 3.11

## 📂 项目目录结构
```text
Geotechnical-AI-Agent/
├── .env.example          # 环境变量示例文件（配置 API Key）
├── app.py                # Web 应用启动入口
├── requirements.txt      # 项目依赖清单
├── data/                 # 数据文件夹
│   ├── raw/              # 原始 PDF 规范文档
│   ├── processed/        # 切分后的 JSON 数据
│   └── vector_db/        # Chroma 向量数据库
├── src/                  # 核心源码
│   ├── tools.py          # Agent 工具集（计算器 & 检索）
│   ├── rag_engine.py     # RAG 检索封装
│   └── config.py         # 配置加载
├── tests/                # 测试脚本
└── archive/              # 历史学习过程代码0

