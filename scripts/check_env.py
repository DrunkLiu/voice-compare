"""环境检查脚本：输出 Python、依赖和 FFmpeg 是否可用。"""

import shutil
import sys

import fastapi
import httpx
import pytest
import uvicorn

# 输出 Python 版本
print(f"Python: {sys.version.split()[0]}")
# 输出核心 Python 依赖版本
print(f"FastAPI: {fastapi.__version__}")
print(f"uvicorn: {uvicorn.__version__}")
print(f"pytest: {pytest.__version__}")
print(f"httpx: {httpx.__version__}")

# 在系统 PATH 中查找 ffmpeg 和 ffprobe
ffmpeg = shutil.which("ffmpeg")
ffprobe = shutil.which("ffprobe")
print(f"ffmpeg: {ffmpeg or 'MISSING'}")
print(f"ffprobe: {ffprobe or 'MISSING'}")
