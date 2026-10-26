import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from src.tools import calculate_excavation_volume, search_specifications

# 加载 .env 里的 API Key
load_dotenv()

# 1. 初始化大模型
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# 2. 准备工具包
tools = [calculate_excavation_volume, search_specifications]

# 3. 设定 Agent 的角色
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个资深的地下工程AI助手。请根据用户的问题，自行决定使用哪个工具来回答。"),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# 4. 创建 Agent
agent = create_agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. 终极测试
print("\n--- 测试1：算量 ---")
agent_executor.invoke({"input": "帮我算一下长30米，宽15米，深4.5米的基坑土方量。"})

print("\n--- 测试2：查规范 ---")
agent_executor.invoke({"input": "地下工程防水等级是怎么划分的？"})