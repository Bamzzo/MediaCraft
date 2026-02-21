# app/context.py
from contextvars import ContextVar

# 定义全局上下文变量
# default={} 防止在非请求环境下导入报错
current_model_config = ContextVar("model_config", default={})
# 👈 新增：用于在不同层级间传递前端上传的图片与视觉模型选择
current_image_data = ContextVar("image_data", default=None)
# 👈 新增：用于在后台隐式传递视频 Base64 数据
current_video_data = ContextVar("video_data", default=None)
current_vision_model = ContextVar("vision_model", default="Qwen2-VL")
