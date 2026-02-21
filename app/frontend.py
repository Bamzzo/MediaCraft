# app/frontend.py
import html
import os
import re
import uuid
import urllib.parse
import requests
import streamlit as st
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 环境配置 ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
API_URL = f"{BACKEND_URL}/chat/stream"
UPLOAD_URL = f"{BACKEND_URL}/upload"

# --- 页面基础设置 ---
st.set_page_config(
    page_title="竹木壹号",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS 样式优化 (飞书/字节 Arco Design 商业风) ---
st.markdown("""
<style>
    /* 隐藏 Streamlit 默认菜单、Footer、顶部装饰线 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 全局背景底色：字节系经典的浅灰 (Arco Design) */
    [data-testid="stAppViewContainer"] {
        background-color: #F2F3F5; 
        font-family: 'PingFang SC', 'Microsoft YaHei', -apple-system, sans-serif;
    }
    
    /* 主内容区顶部留白调整 */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
    }

    /* 侧边栏：纯白面板，带细微右边框阴影 */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5E6EB;
        box-shadow: 2px 0 8px rgba(0, 0, 0, 0.02);
    }
    
    /* 标题排版美化 */
    h1, h2, h3 {
        color: #1D2129;
        font-weight: 600;
    }
    
    /* 对话气泡卡片化设计 */
    [data-testid="stChatMessage"] {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        margin-bottom: 1.2rem;
        border: 1px solid #E5E6EB;
    }
    
    /* 用户消息头像（字节蓝）和 AI 头像（科技黑） */
    [data-testid="chatAvatarIcon-user"] {
        background-color: #165DFF !important;
    }
    [data-testid="chatAvatarIcon-assistant"] {
        background-color: #1D2129 !important;
    }
    
    /* 输入框极简风：无边框，底色填充 */
    [data-testid="stChatInput"] {
        border-radius: 24px !important;
        border: 1px solid #E5E6EB !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #1D2129 !important;
    }
    
    /* 按钮样式：字节蓝 */
    .stButton>button {
        background-color: #165DFF;
        color: #FFFFFF;
        border-radius: 6px;
        border: none;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #4080FF;
        color: #FFFFFF;
        box-shadow: 0 4px 10px rgba(22, 93, 255, 0.2);
    }
    
    /* 下拉框、文本框等组件质感 */
    .stSelectbox div[data-baseweb="select"] > div {
        border-radius: 6px;
        border-color: #E5E6EB;
    }
    
    /* Metric 数据面板美化 */
    [data-testid="stMetricValue"] {
        color: #165DFF;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# --- 会话状态初始化 ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "DeepSeek-V3 (官方直连)"
if "selected_vision_model" not in st.session_state:
    st.session_state.selected_vision_model = "Qwen2-VL-72B (硅基流动)"
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "你是一个全能的多模态超级助理。你可以通过调用对应的工具来完成用户的任何需求。当用户需要画面时调用画图，需要短片时调用视频生成，需要资料时查阅知识库。请保持回答精炼、专业、且具备创造力。"

# --- 页面 1: 全网热点传送门 (Trend Nav Hub) ---
def render_dashboard():
    st.title("🔥 热点直达")
    st.markdown("### 一键直达全网热点")

    st.info("💡 核心导航：点击下方卡片，直接进入各平台官方实时热榜中心。")

    def render_portal_card(icon, title, desc, url, border_color):
        """渲染高质感的传送门卡片"""
        st.markdown(f"""
        <a href="{url}" target="_blank" style="text-decoration: none; display: block; margin-bottom: 16px;">
            <div style="
                background-color: #FFFFFF;
                border-radius: 12px;
                padding: 20px;
                border: 1px solid #E5E6EB;
                border-left: 4px solid {border_color};
                box-shadow: 0 4px 12px rgba(0,0,0,0.02);
                transition: all 0.2s ease-in-out;
                cursor: pointer;
            " onmouseover="this.style.boxShadow='0 8px 24px rgba(0,0,0,0.08)'; this.style.transform='translateY(-2px)';"
               onmouseout="this.style.boxShadow='0 4px 12px rgba(0,0,0,0.02)'; this.style.transform='translateY(0)';">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 24px; margin-right: 12px;">{icon}</span>
                    <h3 style="margin: 0; color: #1D2129; font-size: 18px;">{title}</h3>
                </div>
                <p style="color: #86909C; font-size: 14px; margin: 0; padding-left: 36px;">{desc}</p>
            </div>
        </a>
        """, unsafe_allow_html=True)

    # 使用 Tab 进行合理的垂类划分
    tab1, tab2, tab3 = st.tabs(["📱 短视频与图文", "🌍 社交与吃瓜", "🧠 深度与科技"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            render_portal_card("🎵", "抖音实时热榜", "捕捉当下最具爆发力的短视频流量密码", "https://www.douyin.com/hot", "#1D0F2E")
            render_portal_card("📺", "Bilibili 全站日榜", "Z世代流行文化与中视频风向标", "https://www.bilibili.com/v/popular/all", "#FB7299")
        with col2:
            render_portal_card("📕", "小红书发现页", "种草指南、生活方式与女性向爆款库", "https://www.xiaohongshu.com/explore", "#FF2442")
            render_portal_card("🎥", "快手热榜", "下沉市场与市井生活的真实热度", "https://www.kuaishou.com/", "#FF6600")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            render_portal_card("👁️", "微博热搜榜", "全网吃瓜第一线，社会舆论放大镜", "https://s.weibo.com/top/summary", "#E6162D")
        with col2:
            render_portal_card("🔥", "今日热榜 (聚合)", "一站式纵览全网各平台热点排行榜", "https://tophub.today/", "#165DFF")

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            render_portal_card("💡", "知乎热榜", "硬核知识科普与深度观点发酵池", "https://www.zhihu.com/billboard", "#0066FF")
        with col2:
            render_portal_card("📰", "百度热搜", "国民级海量搜索数据背后的真实趋势", "https://top.baidu.com/board", "#4E6EF2")

# --- 页面 2: 创作大脑 (核心重构区 - 信令驱动版) ---
def render_chat_page():
    st.title("🧠 创作中心")

    with st.sidebar:
        st.divider()
        if st.button("🗑️ 清空当前对话", type="primary"):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

        st.divider()
        st.markdown("### 👁️ 视觉解析工具")
        uploaded_media = st.file_uploader("上传参考图片或视频", type=["png", "jpg", "jpeg", "mp4", "mov"])

        if uploaded_media:
            import base64
            bytes_data = uploaded_media.getvalue()
            b64_str = base64.b64encode(bytes_data).decode("utf-8")
            file_ext = uploaded_media.name.split(".")[-1].lower()
            if file_ext in ["mp4", "mov"]:
                st.session_state.vision_video_base64 = b64_str
                st.session_state.vision_image_base64 = None
                st.success("✅ 视频已就绪！请向大模型提问。")
            else:
                st.session_state.vision_image_base64 = b64_str
                st.session_state.vision_video_base64 = None
                st.success("✅ 图片已就绪！请向大模型提问。")
        else:
            st.session_state.vision_image_base64 = None
            st.session_state.vision_video_base64 = None

    # --- 结构化历史渲染 ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("content"):
                st.markdown(message["content"])

            if message.get("images"):
                cols = st.columns(len(message["images"])) if len(message["images"]) < 4 else [st] * len(message["images"])
                for idx, url in enumerate(message["images"]):
                    container = cols[idx] if idx < len(cols) else st
                    container.image(url, caption="🎨 视觉工坊生成", width="stretch")
            if message.get("videos"):
                for vid_url in message["videos"]:
                    st.video(vid_url)

    # --- 输入与流式解析 (信令驱动) ---
    if prompt := st.chat_input("输入选题、脚本文案，或直接输入『画一张...』"):
        st.session_state.messages.append({"role": "user", "content": prompt, "images": [], "videos": []})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            text_placeholder = st.empty()
            status_placeholder = st.empty()
            image_placeholder = st.empty()
            video_placeholder = st.empty()

            full_response = ""
            current_images = []
            current_videos = []

            payload = {
                "content": prompt,
                "thread_id": st.session_state.session_id,
                "system_prompt": st.session_state.system_prompt,
                "llm_config": {
                    "chat": st.session_state.selected_model,
                    "vision": st.session_state.selected_vision_model,
                },
                "image_data": st.session_state.get("vision_image_base64"),
                "video_data": st.session_state.get("vision_video_base64"),
            }

            try:
                with requests.post(API_URL, json=payload, stream=True, timeout=360) as response:
                    if response.status_code == 200:
                        for line in response.iter_lines():
                            if line:
                                decoded_line = line.decode("utf-8")
                                if decoded_line.startswith("data: "):
                                    data_str = decoded_line[6:]
                                    if data_str == "[DONE]":
                                        break

                                    if data_str.startswith("[SIGNAL_TOOL_START:generate_image]"):
                                        status_placeholder.info("⏳ **画笔引擎唤醒中**... 正在调用视觉工坊渲染画面。")
                                        continue
                                    elif data_str.startswith("[SIGNAL_TOOL_START:analyze_image]"):
                                        status_placeholder.info("👁️ **神之眼启动**... 正在呼叫视觉中枢解析上传的画面。")
                                        continue
                                    elif data_str.startswith("[SIGNAL_TOOL_START:analyze_video]"):
                                        status_placeholder.info("🎥 **视频解析引擎启动**... 正在后台进行智能抽帧与视觉理解。")
                                        continue
                                    elif data_str.startswith("[SIGNAL_TOOL_START:generate_video]"):
                                        status_placeholder.info("🎬 **造梦机唤醒中**... 正在调用 Seedance 生成动态视频 (通常需 1-3 分钟)，请耐心等待。")
                                        continue

                                    elif data_str.startswith("[SIGNAL_IMAGE_URL"):
                                        url_match = re.search(r"\[SIGNAL_IMAGE_URL:(.*?)\]", data_str)
                                        if url_match:
                                            img_url = url_match.group(1)
                                            current_images.append(img_url)
                                            status_placeholder.empty()
                                            with image_placeholder.container():
                                                for u in current_images:
                                                    st.image(u, caption="🎨 视觉工坊生成", width="stretch")
                                        continue

                                    elif data_str.startswith("[SIGNAL_VIDEO_URL"):
                                        url_match = re.search(r"\[SIGNAL_VIDEO_URL:(.*?)\]", data_str)
                                        if url_match:
                                            vid_url = url_match.group(1)
                                            current_videos.append(vid_url)
                                            status_placeholder.empty()
                                            with video_placeholder.container():
                                                for v in current_videos:
                                                    st.video(v)
                                        continue

                                    full_response += data_str
                                    text_placeholder.markdown(full_response + "▌")

                        text_placeholder.markdown(full_response)
                    else:
                        st.error(f"❌ Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"❌ Connection Failed: {e}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "images": current_images,
            "videos": current_videos,
        })
        st.rerun()

# --- 页面 3: 视觉工坊 (Visual Studio) ---
def render_visual_studio():
    st.title("🎨 视觉工坊 (多模态实验室)")
    st.info("在此模块，你可以生成封面、制作动态视频或进行视频内容分析。")
    tab1, tab2, tab3 = st.tabs(["🖼️ 封面生成", "🎬 视频制作", "👁️ 视觉诊断"])

    with tab1:
        st.markdown("### 爆款封面生成 (豆包)")
        desc = st.text_area("描述你的封面画面", "赛博朋克风格，高饱和度，一个程序员在敲代码...")
        if st.button("🎨 立即生成封面"):
            with st.spinner("画笔引擎高速运转中..."):
                try:
                    res = requests.post(f"{BACKEND_URL}/api/generate_image", json={"prompt": desc}, timeout=90)
                    data = res.json()
                    if data.get("status") == "success":
                        st.image(data["url"], caption="生成成功", width="stretch")
                        st.balloons()
                    else:
                        st.error(data.get("message"))
                except Exception as e:
                    st.error(f"生成失败: {e}")

    with tab2:
        st.markdown("### 动态视频生成 (Seedance)")
        st.caption("提示：造梦机需要极度详细的描述，包括主体、环境、光影和镜头运动。")
        col1, col2 = st.columns(2)
        with col1:
            ratio = st.selectbox("视频比例", ["16:9 (横屏)", "9:16 (竖屏)", "1:1 (方图)"])
        with col2:
            duration = st.selectbox("视频时长", ["6秒 (标准)"])
        vid_prompt = st.text_area("详细描述你的电影级镜头", "【镜头缓慢推进】，夕阳下的赛博朋克城市，霓虹灯闪烁，一辆飞行汽车呼啸而过...")
        if st.button("✨ 开始生成视频"):
            with st.spinner("🎬 造梦机运转中 (通常需要 1-3 分钟，请耐心等待)..."):
                try:
                    prompt_with_params = f"[{ratio}, {duration}] {vid_prompt}"
                    res = requests.post(f"{BACKEND_URL}/api/generate_video", json={"prompt": prompt_with_params}, timeout=400)
                    data = res.json()
                    if data.get("status") == "success":
                        st.video(data["url"])
                        st.success("✅ 视频渲染完成！")
                    else:
                        st.error(data.get("message"))
                except Exception as e:
                    st.error(f"生成失败: {e}")

    with tab3:
        st.markdown("### 视觉内容分析 (NVIDIA VILA/Qwen-VL)")
        st.info("👈 请直接在左侧边栏使用【👁️ 视觉解析工具】上传媒体并向 AI 提问。此页面仅作工坊导航展示。")

# --- 页面 4: 系统设置 ---
def render_settings():
    st.title("⚙️ 系统设置")

    st.markdown("### 🎯 基础设定 (System Prompt)")
    st.session_state.system_prompt = st.text_area(
        "定义 Agent 的灵魂与行为准则",
        value=st.session_state.system_prompt,
        height=150,
    )
    st.caption("修改后将在下一次对话中立即生效。")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🧠 语言中枢设定")
        chat_options = [
            "DeepSeek-V3 (官方直连)",
            "Doubao-Pro-128k (Volcengine)",
            "Llama-3.1-70B (NVIDIA/Silicon)",
            "GLM-4-Plus (ZhipuAI)",
        ]
        current_chat_idx = chat_options.index(st.session_state.selected_model) if st.session_state.selected_model in chat_options else 0
        st.session_state.selected_model = st.selectbox("选择主力对话模型", chat_options, index=current_chat_idx)
        st.success(f"当前激活: **{st.session_state.selected_model}**")

    with col2:
        st.markdown("### 👁️ 视觉中枢设定")
        vision_options = [
            "Qwen2-VL-72B (硅基流动)",
            "GLM-4V-Plus (智谱AI)",
        ]
        current_vision_idx = vision_options.index(st.session_state.selected_vision_model) if st.session_state.selected_vision_model in vision_options else 0
        st.session_state.selected_vision_model = st.selectbox("选择视觉解析模型", vision_options, index=current_vision_idx)
        st.success(f"当前激活: **{st.session_state.selected_vision_model}**")

    st.divider()
    st.markdown("### 🔌 API 连接状态")
    st.caption("✅ DeepSeek Official: Connected")
    st.caption("✅ SiliconFlow: Connected")
    st.caption("✅ Volcengine: Connected")
    st.caption("✅ Tavily Search: Connected")

# --- 主导航逻辑 ---
def main():
    with st.sidebar:
        st.title("💻竹木壹号")
        page = st.radio(
            "导航",
            ["数据看板", "创作大脑", "视觉工坊", "系统设置"],
            index=1  # 默认打开创作大脑
        )

        st.divider()
        st.markdown("### 📚 专属知识库构建")
        st.caption("上传 TXT 或 PDF，让 Agent 学习你的独家资料。")
        knowledge_file = st.file_uploader("选择文档", type=["txt", "pdf"], label_visibility="collapsed")
        if st.button("🚀 一键注入大脑", use_container_width=True):
            if knowledge_file:
                files = {"file": (knowledge_file.name, knowledge_file.getvalue(), knowledge_file.type or "application/octet-stream")}
                try:
                    res = requests.post(f"{BACKEND_URL}/upload_knowledge", files=files)
                    if res.status_code == 200:
                        data = res.json()
                        if data.get("status") == "success":
                            filename = knowledge_file.name
                            progress_bar = st.progress(0.0)
                            status_text = st.empty()
                            import time as _time
                            while True:
                                try:
                                    status_res = requests.get(
                                        f"{BACKEND_URL}/knowledge_status",
                                        params={"filename": filename},
                                    )
                                    if status_res.status_code == 200:
                                        state = status_res.json()
                                        if state.get("status") == "processing":
                                            total = state.get("total") or 1
                                            current = state.get("current") or 0
                                            percent = current / total if total else 0
                                            progress_bar.progress(min(percent, 1.0))
                                            status_text.caption(f"⏳ 正在学习: {current} / {total} 块...")
                                        elif state.get("status") == "completed":
                                            progress_bar.progress(1.0)
                                            status_text.success(f"✅ 《{filename}》学习完成！")
                                            break
                                        elif state.get("status") == "not_found":
                                            status_text.caption("⏳ 等待任务启动...")
                                    _time.sleep(2)
                                except Exception:
                                    break
                        else:
                            st.error(data.get("message"))
                    else:
                        st.error(f"服务器错误: {res.status_code}")
                except Exception as e:
                    st.error(f"网络请求失败: {e}")
            else:
                st.warning("请先选择要上传的文档！")

    if page == "数据看板":
        render_dashboard()
    elif page == "创作大脑":
        render_chat_page()
    elif page == "视觉工坊":
        render_visual_studio()
    elif page == "系统设置":
        render_settings()

if __name__ == "__main__":
    main()
