# ⚡ MediaCraft | 竹木壹号

**基于 LangGraph 与 RAG 的多模态 Agent 集成调度系统**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)
![LangGraph](https://img.shields.io/badge/LangGraph-State%20Machine-FF9900)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B)
![Vibe Coding](https://img.shields.io/badge/Paradigm-Vibe%20Coding-8A2BE2)

> **MediaCraft** 是一个专注于“全流程生成”与“底层状态调度”的多模态 AI 创作中枢。
> 本项目基于 `LangGraph` 状态机构建，通过统一的上下文环境，精细调度文本检索、视觉感知、图像生成与视频渲染等多种模型能力，致力于打破工具孤岛，打造一个所见即所得的极客创作工坊。

---

## 💡 开发初衷 (Inspiration & Origin)

作为一个热衷于探索 AI 技术、喜欢在各大社交平台捕捉实时热点与创作灵感的创作者兼开发者，我时常陷入一种效率困境：**在日常的创作中，我不得不频繁地在社交平台（找寻热点）、DeepSeek（进行深度推理与文案构思）以及各类多模态工具（成图成片）之间来回切换。**

这让我不禁思考：能否打破这些工具的孤岛，打造一个集**“全网热点实时了解”、“大模型的协作产出”、“提示词调优辅助”**与**“个人资料库的理解应用”**于一身的创作中枢？

带着这个想法，我在技术平台进行学习探究，接触并学习到了 AI Agent 的理念，并进一步钻研了 `LangChain` 与 `LangGraph` 这两个极具潜力的底层框架。结合我的 Python 编码能力、数据结构与算法基础，以及对各个大模型在创作中的优缺点和真实创作流的理解，在 Gemini 的架构推演与 Cursor 的编码辅助下，**MediaCraft (竹木壹号)** 最终成功开发。

它不仅是我个人对日常创作流“集成化”的深刻理解，也是一次基于 Vibe Coding，将我的创意直觉体现于 AI Agent 工程落地的完整开发尝试。

---

## 🚀 核心亮点 (Highlights)

- **状态机编排**: 基于 LangGraph 实现复杂工作流的确定性流转，超越传统的黑盒 AgentExecutor。
- **并发状态隔离**: 运用 Python `ContextVar` 实现跨请求的多模态上下文（图片/视频 Base64）数据绝对安全隔离。
- **信令通信协议**: 设计自定义 `SIGNAL` 信令拦截机制，实现大模型底层推理与前端多模态 UI 渲染的无缝流式解耦。
- **工业级 RAG**: Chroma 向量海选 + BGE-Reranker 语义精排，并结合算法基础实现了带随机抖动的指数退避机制以抵抗 API 限流。
- **全栈多模态闭环**: 深度集成 OpenCV 视频本地抽帧、多模态视觉理解 (Qwen-VL / GLM-4V) 与动态媒体流生成 (Flux / Seedance)。

---

## 🏗️ 系统架构 (Architecture)

```text
User / Creator
  │
  ├──► [Streamlit UI] 多模态前端工作站 & 全局 Prompt 调优面板
  │      │
  │      ▼ (Server-Sent Events 流式通信 & 自定义 SIGNAL 协议)
  │
  ├──► [FastAPI Backend] 高并发异步底层 (ContextVar 状态隔离)
  │      │
  │      ▼
  └──► [LangGraph State Machine] 核心调度大脑
         │
         ├──► 📚 RAG Engine       (个人资料库：Chroma + Reranker)
         ├──► 👁️ Vision Central   (视觉理解：OpenCV + Qwen2-VL/GLM-4V)
         ├──► 🎨 Image Generator  (视觉产出：Flux.1 / 豆包画图)
         ├──► 🎬 Video Generator  (动态产出：火山引擎 Seedance 造梦机)
         └──► 🌐 Web Search       (实时热点：Tavily 全网搜索)
```

---

## ✨ 核心工程实现 (Core Implementation)

### 1. 知识库的高可用吞噬 (Robust RAG System)
为实现“个人资料库的理解应用”，系统支持长文本（PDF/TXT）的异步向量化。在切片入库过程中，结合算法思维实现了**指数退避重试算法 (Exponential Backoff)**，有效应对大文档并行处理时频繁触发的 LLM API `429 Too Many Requests` 限流崩溃问题，保障了知识库构建的健壮性。

### 2. 跨模态数据的无污染穿透 (Context Management)
针对 ASGI 异步服务器下的多用户并发场景，极其克制地使用了 `ContextVar`。成功实现了用户上传的视频/图片数据在“全局上下文 -> LangGraph 节点 -> 具体 Tool 工具”链路中的精准传递，彻底杜绝了高并发环境下的数据串线。

### 3. 多模态生成的信令驱动 (Signal-Driven UI)
首创 `[SIGNAL_TOOL_START:xxx]` 前后端拦截协议。造梦机或画图引擎成功运行后，向大模型隐式返回系统长链接，严令大模型以自然语言沟通而不输出 Markdown。前端截获信令并异步渲染图像/视频组件，实现了极度丝滑的创作交互体验。

---

## ⚙️ 快速开始 (Quick Start)

### 1. 环境准备
建议在 Python 3.10+ 的独立虚拟环境中运行本系统。
```bash
git clone [https://github.com/Bamzzo/MediaCraft.git](https://github.com/Bamzzo/MediaCraft.git)
cd MediaCraft

# 创建并激活虚拟环境
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# 安装核心依赖
pip install -r requirements.txt
```

### 2. API 密钥配置
复制环境配置模板，并填入对应的大模型密钥：
```bash
cp .env.example .env
```
> **依赖说明**：本项目模型推理高度依赖 SiliconFlow（硅基流动）、Volcengine（火山引擎）、ZhipuAI（智谱）等平台的原生 API，请确保账户已配置充足额度及对应模型权限（如 Seedance）。

### 3. 系统启动
请在项目根目录下，**开启两个独立的终端窗口**，分别唤醒大脑与前端：

**Terminal 1: 启动高并发后端中枢 (FastAPI)**
```bash
python -m app.main
```

**Terminal 2: 启动多模态创作台 (Streamlit)**
```bash
streamlit run app/frontend.py
```
启动无误后，浏览器将自动访问 `http://localhost:8501`。

---

## 🎬 创作流演示 (Demo Workflow)

1. **热点洞察**：在【数据看板】一键直达各平台热榜。
2. **资料外脑**：向侧边栏传入专属文档，通过 RAG 引擎瞬间完成知识对齐。
3. **视觉感知**：上传本地参考短片，系统后台通过 OpenCV 抽帧并交由视觉大模型解析分镜细节。
4. **协同产出**：通过设定全局 System Prompt 调优 Agent 人格，输入一句话指令，LangGraph 将自动规划路径：*脚本重写 -> 调用 Flux 生成分镜 -> 呼叫 Seedance 渲染最终视频*。

---

## 📄 许可证 (License)

本项目采用 [MIT License](LICENSE) 开源许可证。欢迎自由使用与二次开发。

---

## 👤 作者 (Author)

**竹木 (Bamzzo)** | CUC DMT
- 正在探索：Agent / Prompt工程 / Vibe Coding / UE / Unity / 喜欢探索学习一些有趣前沿的创作和开发知识，做一些有趣好用的东西！
- GitHub: [@Bamzzo](https://github.com/Bamzzo)
