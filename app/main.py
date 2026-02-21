# app/main.py
import os
import re
import uvicorn
from dotenv import load_dotenv  # 👈 引入 dotenv

# ⚠️ 极其关键：在所有代码运行前加载环境变量！
load_dotenv() 

from fastapi import BackgroundTasks, FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import Optional

from langchain_core.messages import HumanMessage
from sse_starlette.sse import EventSourceResponse

from app.agent import app_graph
from app.context import current_model_config, current_image_data, current_video_data, current_vision_model

app = FastAPI(title="ByteCreator Backend")

# --- 视觉工坊专用直连 API ---
class ImageRequest(BaseModel):
    prompt: str


class VideoRequest(BaseModel):
    prompt: str


@app.post("/api/generate_image")
async def api_generate_image(req: ImageRequest):
    from app.tools import generate_image

    result = generate_image.invoke({"prompt": req.prompt})
    url_match = re.search(r"\[System Hidden URL:\s*(https?://[^\s\]]+)\]", str(result))
    if url_match:
        return {"status": "success", "url": url_match.group(1)}
    return {"status": "error", "message": result}


@app.post("/api/generate_video")
async def api_generate_video(req: VideoRequest):
    from app.tools import generate_video

    result = generate_video.invoke({"prompt": req.prompt})
    url_match = re.search(r"\[System Hidden Video URL:\s*(https?://[^\s\]]+)\]", str(result))
    if url_match:
        return {"status": "success", "url": url_match.group(1)}
    return {"status": "error", "message": result}


# --- 数据模型定义 ---
class ModelConfig(BaseModel):
    chat: str = "DeepSeek-V3 (SiliconFlow)"
    image: Optional[str] = "Flux.1"
    vision: Optional[str] = "Qwen-VL"

class ChatRequest(BaseModel):
    content: str
    thread_id: str
    system_prompt: str = "你是一个智能助手。"
    llm_config: Optional[ModelConfig] = None
    image_data: Optional[str] = None
    video_data: Optional[str] = None  # 👈 新增视频 Base64 接收字段


# --- 接口定义 ---
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式对话接口，支持多模型切换
    """
    llm_config = request.llm_config or ModelConfig()

    async def event_generator():
        token_config = current_model_config.set({"chat": llm_config.chat, "vision": llm_config.vision})
        token_img = current_image_data.set(request.image_data)
        token_vid = current_video_data.set(request.video_data)
        token_vision = current_vision_model.set(llm_config.vision)

        try:
            user_text = request.content
            if request.image_data:
                user_text = f"【系统提示：用户在本次对话中附带上传了一张图片。请立刻调用 'analyze_uploaded_image' 工具进行解析。】\n\n用户输入：{request.content}"
            elif request.video_data:
                user_text = f"【系统提示：用户在本次对话中附带上传了一段视频。请立刻调用 'analyze_uploaded_video' 工具进行抽帧与解析。】\n\n用户输入：{request.content}"

            inputs = {"messages": [HumanMessage(content=user_text)]}

            config = {
                "configurable": {
                    "thread_id": request.thread_id,
                    "selected_chat_model": llm_config.chat,
                    "system_prompt": request.system_prompt,
                }
            }

            async for event in app_graph.astream_events(inputs, config=config, version="v1"):
                kind = event["event"]

                if kind == "on_tool_start":
                    if event.get("name") == "generate_image":
                        yield {"data": "[SIGNAL_TOOL_START:generate_image]"}
                    elif event.get("name") == "analyze_uploaded_image":
                        yield {"data": "[SIGNAL_TOOL_START:analyze_image]"}
                    elif event.get("name") == "analyze_uploaded_video":
                        yield {"data": "[SIGNAL_TOOL_START:analyze_video]"}
                    elif event.get("name") == "generate_video":
                        yield {"data": "[SIGNAL_TOOL_START:generate_video]"}

                elif kind == "on_tool_end":
                    tool_name = event.get("name")
                    output = event["data"].get("output")
                    if not output:
                        pass
                    elif tool_name == "generate_image":
                        url_match = re.search(r'\[System Hidden URL:\s*(https?://[^\s\]]+)\]', str(output))
                        if url_match:
                            yield {"data": f"[SIGNAL_IMAGE_URL:{url_match.group(1)}]"}
                    elif tool_name == "generate_video":
                        url_match = re.search(r'\[System Hidden Video URL:\s*(https?://[^\s\]]+)\]', str(output))
                        if url_match:
                            yield {"data": f"[SIGNAL_VIDEO_URL:{url_match.group(1)}]"}

                # 💬 常规模型文本流
                elif kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        yield {"data": chunk.content}

        except Exception as e:
            print(f"❌ Error in stream: {e}")
            yield {"data": f"[ERROR] {str(e)}"}
        finally:
            yield {"data": "[DONE]"}
            current_model_config.reset(token_config)
            current_image_data.reset(token_img)
            current_video_data.reset(token_vid)
            current_vision_model.reset(token_vision)

    return EventSourceResponse(event_generator())


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename, "status": "success"}


@app.get("/knowledge_status")
async def get_knowledge_status(filename: str):
    from app.rag import knowledge_progress

    return knowledge_progress.get(filename, {"status": "not_found"})


@app.post("/upload_knowledge")
async def upload_knowledge(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    接收前端上传的文档，解析出纯文本后送入 RAG 知识库（后台异步入库，立即返回）
    """
    from app.rag import add_to_knowledge_base

    content = ""
    try:
        if file.filename.lower().endswith(".txt"):
            raw_bytes = await file.read()
            try:
                content = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    content = raw_bytes.decode("gb18030")
                except Exception as encode_err:
                    return {"status": "error", "message": f"❌ 文本编码不支持。请将 TXT 另存为 UTF-8 格式后重试。报错: {encode_err}"}
        elif file.filename.lower().endswith(".pdf"):
            import fitz

            raw_bytes = await file.read()
            with fitz.open(stream=raw_bytes, filetype="pdf") as pdf_document:
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    text = page.get_text()
                    if text:
                        content += text + "\n"
        else:
            return {"status": "error", "message": "❌ 仅支持 TXT 或 PDF 格式的文档"}

        if not content.strip():
            return {"status": "error", "message": "❌ 文件内容为空或无法解析"}

        background_tasks.add_task(add_to_knowledge_base, content, file.filename)
        approx_chunks = max(1, len(content) // 500)
        return {
            "status": "success",
            "message": f"🚀 文件已接收！共计约 {approx_chunks} 个知识块正在后台异步注入大脑，请稍等片刻。",
        }

    except Exception as e:
        print(f"解析文档异常: {e}")
        return {"status": "error", "message": f"❌ 解析失败: {str(e)}"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)