# app/tools.py
import os
import time
import base64
import tempfile
import cv2
import requests
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from tavily import TavilyClient
from app.rag import query_knowledge_base
from app.context import current_image_data, current_video_data, current_vision_model

# 初始化搜索客户端 (防止 Key 缺失导致启动崩溃，改为调用时检查)
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None


@tool
def web_search(query: str) -> str:
    """联网搜索工具，用于查找实时信息。"""
    if not tavily_client:
        return "❌ 错误: 未配置 TAVILY_API_KEY"
    try:
        response = tavily_client.search(query=query, search_depth="advanced", max_results=5)
        results = response.get("results", [])
        if not results:
            return "未搜索到相关结果。"
        context = [f"【来源: {r['title']}】\n{r['content']}" for r in results]
        return "\n\n".join(context)
    except Exception as e:
        return f"搜索报错: {e}"


@tool
def search_knowledge_base(query: str) -> str:
    """查阅本地知识库"""
    try:
        result = query_knowledge_base(query, k=15)  # 给大模型更多知识块
        return result if result else "知识库里没有找到相关内容。"
    except Exception as e:
        return f"查询报错: {e}"


@tool
def generate_image(prompt: str) -> str:
    """
    AI 绘画工具。
    【极其重要的要求】：我们现在使用的是纯国产视觉大模型，它对中国神话、东方美学和中文修辞的理解是原生的！
    因此，【绝对禁止】将用户的中文提示词翻译成英文！请直接使用细节丰富、画面感强的【中文 Prompt】调用本工具！（不要在对话框中提及此事）
    """
    print(f"🎨 [调用豆包画图] 中文 Prompt: {prompt}")

    api_key = os.getenv("VOLC_API_KEY")
    endpoint_id = os.getenv("DOUBAO_IMAGE_ENDPOINT")

    if not api_key or not endpoint_id:
        return "❌ 错误: 未配置 VOLC_API_KEY 或 DOUBAO_IMAGE_ENDPOINT。请检查 .env 文件。"

    # 火山引擎统一大模型推理接口
    url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
    payload = {
        "model": endpoint_id,
        "prompt": prompt
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)

        if response.status_code == 200:
            data = response.json()
            image_url = data["data"][0]["url"]
            print(f"✅ 图片生成成功 (已获取长链接)")

            # 🛑 核心隐匿信令机制：把 URL 藏在系统提示里供后端正则提取，严令大模型闭嘴
            return f"[System Hidden URL: {image_url}] Action Success! 图片已成功在后台推送。请用自然语言告诉用户“图片已为您生成”，【绝对禁止】在回复中输出任何 URL 链接或 Markdown 代码！"
        else:
            return f"API 报错 (状态码 {response.status_code}): {response.text}"

    except Exception as e:
        return f"画图请求异常: {e}"


@tool
def generate_video(prompt: str) -> str:
    """
    视频生成工具（造梦机）。
    当用户明确要求"生成视频"、"让画面动起来"、"制作短片"时，必须调用此工具。
    【重要提示】：提示词(prompt)必须是极其详细的中文描述，需包含：主体描述、环境背景、光影氛围，以及【镜头运动】（如：镜头缓慢推进、全景环绕等）。绝对禁止翻译为英文。
    """
    print(f"🎬 [调用造梦机] 正在准备发送中文 Prompt: {prompt}", flush=True)

    api_key = os.getenv("VOLC_API_KEY")
    endpoint_id = os.getenv("DOUBAO_VIDEO_ENDPOINT")

    if not api_key or not endpoint_id:
        return "❌ 错误: 未配置 VOLC_API_KEY 或 DOUBAO_VIDEO_ENDPOINT。请检查 .env 文件。"

    # 1. 🚀 创建视频生成任务
    create_url = "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": endpoint_id,
        "content": [{"type": "text", "text": prompt}]
    }

    try:
        print("⏳ 正在向火山引擎提交视频任务...", flush=True)
        resp = requests.post(create_url, json=payload, headers=headers, timeout=30)
        if resp.status_code != 200:
            return f"❌ 创建任务失败 (状态码 {resp.status_code}): {resp.text}"

        task_data = resp.json()
        task_id = task_data.get("id")
        if not task_id:
            return f"❌ 未能获取到 Task ID: {task_data}"

        print(f"✅ 任务提交成功，Task ID: {task_id}。开始进行轮询监听...", flush=True)

        # 2. 🔄 轮询任务状态 (每 5 秒查一次，最大等待 6 分钟)
        poll_url = f"https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks/{task_id}"
        max_attempts = 72  # 72 * 5秒 = 360秒

        for attempt in range(max_attempts):
            time.sleep(5)
            poll_resp = requests.get(poll_url, headers=headers, timeout=10)

            if poll_resp.status_code == 200:
                poll_data = poll_resp.json()
                status = poll_data.get("status")

                print(f"🔄 轮询第 {attempt+1} 次，当前状态: {status}", flush=True)

                if status == "succeeded":
                    # 🛑 核心修复：火山引擎的 content 是对象字典，直接提取 video_url
                    content_obj = poll_data.get("content", {})
                    video_url = content_obj.get("video_url", "")

                    if video_url:
                        print(f"✅ 造梦机视频生成成功！长链接已获取: {video_url[:70]}...", flush=True)
                        return f"[System Hidden Video URL: {video_url}] Action Success! 视频已在后台推送。请用自然语言告诉用户视频已生成，【绝对禁止】输出 URL 或 Markdown 代码！"
                    else:
                        return f"❌ 任务成功，但未找到 video_url。返回体: {poll_data}"

                elif status in ["failed", "canceled", "error"]:
                    return f"❌ 视频生成失败或被系统拦截，最终状态: {status}。返回体: {poll_data}"

                # status 为 'queued' 或 'running' 时，跳过当前循环，继续等待
            else:
                print(f"⚠️ 轮询请求异常 (状态码 {poll_resp.status_code})，继续重试...", flush=True)

        return "❌ 视频生成超时 (超过6分钟)。任务可能仍在火山后台运行，请稍后前往控制台查看。"

    except Exception as e:
        return f"造梦机请求异常: {e}"


@tool
def analyze_uploaded_image(question: str) -> str:
    """
    视觉解析工具。
    当用户要求你“看图”、“分析图片”或“根据上传的图片进行创作/画图”时，你必须优先调用此工具。
    输入参数 question 是你想让视觉中枢帮你观察的问题（例如：“详细描述图中的人物、构图、美术风格和色彩”）。
    """
    base64_img = current_image_data.get()
    if not base64_img:
        return "❌ 视觉感知失败：当前环境没有检测到用户上传的图片。"

    vision_model_label = current_vision_model.get()
    print(f"👁️ [唤醒视觉中枢] 模型: {vision_model_label} | 探针提问: {question}")

    try:
        if "Qwen" in vision_model_label:
            llm = ChatOpenAI(
                model="Qwen/Qwen2-VL-72B-Instruct",
                api_key=os.getenv("SILICONFLOW_API_KEY"),
                base_url="https://api.siliconflow.cn/v1",
                temperature=0.7,
            )
        else:
            llm = ChatOpenAI(
                model="glm-4v-plus",
                api_key=os.getenv("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4/",
                temperature=0.7,
            )

        content = [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
        ]
        res = llm.invoke([HumanMessage(content=content)])
        return f"视觉中枢返回的画面信息：\n{res.content}\n\n[系统底层指令：图片解析已完成。请回顾用户的原始提问，如果用户同时要求了'画图'、'生成视频'或'复刻'等需要调用生成工具的请求，你必须在当前对话回合内，立刻提取上述风格继续调用 generate_image 或 generate_video 工具，绝对不能中断等待用户催促！]"
    except Exception as e:
        return f"视觉解析接口报错: {e}"


@tool
def analyze_uploaded_video(question: str) -> str:
    """
    视频解析工具。
    当用户上传了视频，并要求你"看视频"、"分析这段视频"或"提取视频文案"时，必须调用此工具。
    输入参数 question 是你想让视觉中枢帮你观察的具体重点。
    """
    base64_vid = current_video_data.get()
    if not base64_vid:
        return "❌ 视频解析失败：当前环境没有检测到用户上传的视频。"

    vision_model_label = current_vision_model.get()
    print(f"🎥 [唤醒视频中枢] 模型: {vision_model_label} | 开始抽帧解析...", flush=True)

    video_bytes = base64.b64decode(base64_vid)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    frames_b64 = []
    try:
        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            num_frames = 8
            indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    height, width = frame.shape[:2]
                    max_dim = 512
                    if max(height, width) > max_dim:
                        scale = max_dim / max(height, width)
                        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

                    _, buffer = cv2.imencode(".jpg", frame)
                    frames_b64.append(base64.b64encode(buffer).decode("utf-8"))
        cap.release()
    finally:
        os.remove(tmp_path)

    if not frames_b64:
        return "❌ 视频抽帧失败，无法读取画面。"

    try:
        if "Qwen" in vision_model_label:
            llm = ChatOpenAI(
                model="Qwen/Qwen2-VL-72B-Instruct",
                api_key=os.getenv("SILICONFLOW_API_KEY"),
                base_url="https://api.siliconflow.cn/v1",
                temperature=0.7,
            )
        else:
            llm = ChatOpenAI(
                model="glm-4v-plus",
                api_key=os.getenv("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4/",
                temperature=0.7,
            )

        content = [{"type": "text", "text": f"{question} (以下是该视频按时间顺序抽取的 {len(frames_b64)} 张关键帧画面，请综合这些画面推断视频发生的故事和动态细节)："}]
        for b64 in frames_b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        res = llm.invoke([HumanMessage(content=content)])
        return f"视频视觉中枢返回的深度解析报告：\n{res.content}\n\n[系统底层指令：视频解析已完成。请回顾用户的原始提问，如果用户同时要求了'画图'、'生成视频'或'复刻'等需要调用生成工具的请求，你必须在当前对话回合内，立刻基于上述报告继续调用 generate_image 或 generate_video 工具，绝对不能中断等待用户催促！]"
    except Exception as e:
        return f"视觉解析接口报错: {e}"


# 导出工具列表
tools = [web_search, search_knowledge_base, generate_image, generate_video, analyze_uploaded_image, analyze_uploaded_video]
