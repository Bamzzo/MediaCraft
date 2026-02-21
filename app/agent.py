# app/agent.py
import os
import operator
from typing import Annotated, Sequence, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from app.tools import tools


# --- 🏭 Model Factory (核心工厂) ---
def get_llm(model_label: str):
    """根据前端传来的标签，返回对应的 LLM 实例"""

    print(f"🏭 初始化模型: {model_label}")

    # 1. 官方原生 DeepSeek (直连配置)
    if "DeepSeek" in model_label:
        return ChatOpenAI(
            model="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            temperature=0.7,
            streaming=True,
        )

    # 2. Llama (NVIDIA 官方 NIM 直连，满血 70B)
    elif "Llama" in model_label:
        return ChatOpenAI(
            model="meta/llama-3.1-70b-instruct",
            api_key=os.getenv("NVIDIA_API_KEY"),
            base_url="https://integrate.api.nvidia.com/v1",
            temperature=0.6,
            streaming=True,
        )

    # 2. Volcengine (字节跳动 - 豆包)
    elif "Doubao" in model_label:
        endpoint_id = os.getenv("DOUBAO_LLM_ENDPOINT")
        if not endpoint_id:
            print("⚠️ 警告: 未配置 DOUBAO_LLM_ENDPOINT，回退到 DeepSeek")
            return get_llm("DeepSeek")

        return ChatOpenAI(
            model=endpoint_id,
            api_key=os.getenv("VOLC_API_KEY"),
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            temperature=0.7,
            streaming=True,
        )

    # 3. ZhipuAI (GLM-4-Plus)
    elif "GLM" in model_label:
        return ChatOpenAI(
            model="glm-4-plus",
            api_key=os.getenv("ZHIPU_API_KEY"),
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            temperature=0.7,
            streaming=True,
        )

    # --- 👁️ 神之眼：视觉解析大模型 ---
    elif "Qwen2-VL" in model_label:
        return ChatOpenAI(
            model="Qwen/Qwen2-VL-72B-Instruct",
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url="https://api.siliconflow.cn/v1",
            temperature=0.7,
            streaming=True,
        )

    elif "GLM-4V" in model_label:
        return ChatOpenAI(
            model="glm-4v-plus",
            api_key=os.getenv("ZHIPU_API_KEY"),
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            temperature=0.7,
            streaming=True,
        )

    elif "Qwen" in model_label:
        return ChatOpenAI(
            model="Qwen/Qwen2.5-72B-Instruct",
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url="https://api.siliconflow.cn/v1",
            temperature=0.7,
            streaming=True,
        )

    # 默认兜底
    print(f"⚠️ 未知模型标签 [{model_label}]，降级使用 DeepSeek-V3")
    return get_llm("DeepSeek")


# --- State 定义 ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# --- Nodes (节点逻辑) ---
def call_model(state: AgentState, config: RunnableConfig):
    messages = state["messages"]

    configurable = config.get("configurable", {})
    selected_chat_model = configurable.get("selected_chat_model", "DeepSeek")
    system_prompt_text = configurable.get("system_prompt", "你是一个智能助手。")

    # 大脑永远是文本模型，图片通过 analyze_uploaded_image 工具查看
    identity_prompt = f"\n\n[系统指令：你是基于 {selected_chat_model} 驱动的核心大脑。如果用户附带了图片或视频，你必须分别调用 analyze_uploaded_image 或 analyze_uploaded_video 工具来进行视觉感知。]"
    sys_msg = SystemMessage(content=system_prompt_text + identity_prompt)

    prompt_messages = [sys_msg] + messages
    llm = get_llm(selected_chat_model)
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(prompt_messages)

    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


# --- Graph 构建 ---
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END},
)
workflow.add_edge("tools", "agent")

# 🧠 植入海马体 (MemorySaver)
memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)
