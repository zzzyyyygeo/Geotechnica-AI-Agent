import os
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from src.tools import calculate_excavation_volume, search_specifications

load_dotenv()

# 1. 初始化大模型
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# 2. 准备工具包
tools = [calculate_excavation_volume, search_specifications]

# 3. 设定 Agent 的提示词
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个资深的地下工程AI助手。你可以使用工具来查阅规范和计算工程量。请根据用户问题自行决定使用哪个工具。回答要专业、条理清晰。"),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# 4. 创建 Agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False) # 网页端关掉 verbose 屏蔽后台日志

# 5. 网页端的对话函数
def chat_with_agent(message, history):
    try:
        # Agent 自动判断调用哪个工具并生成最终回答
        response = agent_executor.invoke({"input": message})
        return response["output"]
    except Exception as e:
        return f"发生错误：{e}"

# 6. 搭建 Gradio 界面
demo = gr.ChatInterface(
    fn=chat_with_agent,
    title="🏗️ 地下工程智能 Agent 助手",
    description="不仅能查规范，还能帮你算土方量。试试问我：'帮我算一下长30宽15深4.5的土方量，并查一下防水等级'",
    examples=[
        "地下工程防水等级分为几级？",
        "帮我算一下长50米，宽20米，深6米的基坑土方量。"
    ]
)

if __name__ == "__main__":
    demo.launch()