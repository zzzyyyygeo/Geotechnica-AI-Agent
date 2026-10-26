# -*- coding: utf-8 -*-
"""
Day6 交付物：把 API Key 从代码里拿出来（配置管理）

为什么要有这个文件：
    你之前的 key 直接写在 day01.py / day02_chat.py/key.py / day04 / day05 里，
    一共 6 个文件。问题有两个：
      1. 想换 Key 要改 6 个地方
      2. 文件一旦发给别人（GitHub、面试官、同学），Key 就泄露了

正确的做法（按优先级从高到低找 Key）：
    1. 环境变量 DEEPSEEK_API_KEY
       —— 适合临时用、适合部署到服务器
    2. 项目目录下的 .env 文件（本项目用这个）
       —— 适合本地长期用，文件被 .gitignore 忽略，不会被提交
    3. 都没有 -> 给出清楚的中文提示，而不是抛一个看不懂的报错

这就是「配置管理」：代码里不出现任何密钥，密钥只存在于环境里。
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(_HERE)                 # 项目根目录（src/ 的上一级）
ENV_FILE = os.path.join(PROJECT_DIR, ".env")
EXAMPLE_FILE = os.path.join(PROJECT_DIR, ".env.example")

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"


def load_env_file(path=ENV_FILE):
    """.env 文件格式：一行一个 KEY=VALUE，以 # 开头的是注释。"""
    if not os.path.exists(path):
        return {}
    conf = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            conf[k.strip()] = v.strip().strip('"').strip("'")
    return conf


def get_api_key(required=True):
    """按优先级取 API Key：环境变量 -> .env 文件 -> 报错提示。"""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key.strip()

    conf = load_env_file()
    key = conf.get("DEEPSEEK_API_KEY")
    if key and not key.startswith("sk-替换"):
        return key.strip()

    if not required:
        return None

    raise SystemExit(f"""
❌ 没有找到 API Key。请二选一：

【方式一】写进 .env 文件（推荐，一次配好长期有效）
    1. 用记事本打开（没有就新建）：{ENV_FILE}
    2. 写入这一行（把 sk-xxx 换成你真实的 Key）：
           DEEPSEEK_API_KEY=sk-xxx
    3. 保存，重新运行脚本即可

【方式二】只对当前终端窗口生效
    $env:DEEPSEEK_API_KEY = "sk-xxx"

🔑 Key 去哪里拿：https://platform.deepseek.com/api_keys
⚠️  .env 文件已经被 .gitignore 忽略，不会误传到 GitHub。
""")


def get_base_url():
    return (os.environ.get("DEEPSEEK_BASE_URL")
            or load_env_file().get("DEEPSEEK_BASE_URL")
            or DEFAULT_BASE_URL)


def get_model():
    return (os.environ.get("DEEPSEEK_MODEL")
            or load_env_file().get("DEEPSEEK_MODEL")
            or DEFAULT_MODEL)


def get_llm_provider():
    """用哪个客户端调大模型：'openai'（httpx）还是 'urllib'（标准库）。

    背景：这台机器上出现过 httpx 请求 120 秒超时、而标准库 urllib 3.6 秒成功的
    现象（同一个 Key、同一个网址）。这类问题出在网络中间设备对 TLS 指纹或
    HTTP/2 协商的拦截上，换客户端库就能绕过去。
    切换方式：在 .env 里加一行  LLM_PROVIDER=urllib
    """
    v = (os.environ.get("LLM_PROVIDER")
         or load_env_file().get("LLM_PROVIDER")
         or "openai")
    return v.strip().lower()


def get_llm(timeout=180.0, max_retries=2):
    """统一的取客户端入口：根据配置自动选用 openai 或标准库实现。

    两个返回对象的调用方式一致：
        client.chat_completions.create(model=..., messages=[...], temperature=...)
    """
    if get_llm_provider() == "urllib":
        from .llm_util import SimpleLLM
        return SimpleLLM(timeout=timeout, max_retries=max_retries)
    return get_client(timeout=timeout, max_retries=max_retries)


def get_client(timeout=180.0, max_retries=2, http2=False):
    """一行拿到配好的 DeepSeek 客户端（最常用的入口）。

    ⚠️ 为什么要显式设 timeout（重点，你卡住就是踩了这个坑）：
        openai 库默认「不超时」——网络一旦不通，程序会永远卡在
        `ssl.py` 的 read() 上，什么也不打印，看起来像死机。
        （你按 Ctrl+C 后看到的 KeyboardInterrupt 堆栈就是证据）

    timeout 分两段设，这个区分很重要：
        connect=15  秒：连不上服务器要快速失败，别干等
        read=180    秒：大模型生成本来就慢，读数据要给足时间
      （如果整个 timeout 只给 60 秒，长 Prompt 生成很容易被误判成"失败"）

    为什么要显式关掉 HTTP/2：
        httpx 默认会尝试 HTTP/2。部分网络环境下 HTTP/2 协商会挂起
        （表现为请求发出去、永远等不到响应），强制 HTTP/1.1 更稳。
    """
    import httpx
    from openai import OpenAI

    http_client = httpx.Client(
        timeout=httpx.Timeout(timeout, connect=15.0),
        http2=http2,
        trust_env=False,        # 忽略系统代理，避免被坏代理拖住
    )
    return OpenAI(api_key=get_api_key(), base_url=get_base_url(),
                  timeout=timeout, max_retries=max_retries,
                  http_client=http_client)


if __name__ == "__main__":
    # 自测：检查配置是否就绪（不会打印出完整 Key，只显示头尾）
    from text_cleaner import setup_utf8_stdout
    setup_utf8_stdout()
    print(f".env 文件位置     : {ENV_FILE}")
    print(f".env 文件存在     : {os.path.exists(ENV_FILE)}")
    print(f"base_url          : {get_base_url()}")
    print(f"model             : {get_model()}")
    key = get_api_key(required=False)
    if key:
        print(f"API Key           : {key[:6]}...{key[-4:]}  ✅ 已配置")
    else:
        print("API Key           : ❌ 未配置")
        print(f"\n请新建/编辑这个文件：{ENV_FILE}")
        print("内容写一行：DEEPSEEK_API_KEY=sk-你的key")
