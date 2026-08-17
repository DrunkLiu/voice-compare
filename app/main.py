"""FastAPI 后端入口：提供页面、健康检查、上传和文件列表接口。"""

import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import get_settings
from app.media_validation import (
    MediaValidationError,
    is_image_bytes,
    validate_media_file,
)
from app.storage import insert_file, list_files as list_stored_files

settings = get_settings()
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

UPLOAD_DIR = settings.upload_dir
STATIC_DIR = settings.static_dir
MAX_UPLOAD_BYTES = settings.max_upload_size_mb * 1024 * 1024
CHUNK_SIZE = 1024 * 1024


class LocalUploadRequest(BaseModel):
    """本地路径登记请求体。"""

    path: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时创建必要目录。"""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title=settings.app_name, version=settings.version, lifespan=lifespan)
# 把 /static 路径映射到 static 目录，供浏览器加载静态资源
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 跨域来源由配置控制，部署到正式域名时无需改代码
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _safe_saved_name(original_name: str) -> str:
    """生成磁盘安全文件名：随机前缀 + 安全词干 + 安全扩展名。"""
    stem = Path(original_name).stem
    extension = Path(original_name).suffix.lower()
    safe_stem = re.sub(r"[^A-Za-z0-9_-]", "_", stem)[:80] or "file"
    safe_extension = re.sub(r"[^A-Za-z0-9.]", "", extension)[:10]
    return f"{uuid4().hex}_{safe_stem}{safe_extension}"


@app.get("/", include_in_schema=False)
def index():
    """根路径返回上传页面。"""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    """健康检查接口，用于确认服务是否正常运行。"""
    return {"status": "ok", "version": settings.version}


@app.post("/api/upload")
def upload_file(request: Request, file: UploadFile = File(...)):
    """接收上传文件，校验格式和大小后保存到 uploads 目录。"""
    original_name = Path(file.filename or "unknown").name
    saved_name = _safe_saved_name(original_name)
    target = UPLOAD_DIR / saved_name
    size = 0
    saved = False

    # 根据 Content-Length 提前拒绝超大文件，避免浪费带宽
    content_length = request.headers.get("content-length")
    if (
        content_length
        and content_length.isdigit()
        and int(content_length) > MAX_UPLOAD_BYTES
    ):
        raise HTTPException(
            status_code=413,
            detail=f"文件超过大小限制（{settings.max_upload_size_mb}MB）",
        )

    try:
        # 先读第一块做文件头检查，图片/动图尽早拦截
        first_chunk = file.file.read(CHUNK_SIZE)
        if not first_chunk:
            raise HTTPException(status_code=400, detail="上传文件为空")
        if is_image_bytes(first_chunk):
            raise HTTPException(
                status_code=400,
                detail="不支持图片、动图或 Live Photo，请上传音频或视频",
            )

        size += len(first_chunk)
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"文件超过大小限制（{settings.max_upload_size_mb}MB）",
            )

        # 分块写入磁盘，避免大文件一次性读入内存
        with target.open("wb") as buffer:
            buffer.write(first_chunk)
            while chunk := file.file.read(CHUNK_SIZE):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件超过大小限制（{settings.max_upload_size_mb}MB）",
                    )
                buffer.write(chunk)

        # 完整落盘后，用 filetype + ffprobe 做最终校验
        validate_media_file(target, probe=settings.ffprobe_path)
        record = insert_file(
            original_name=original_name,
            stored_path=target,
            source_type="upload",
            size=size,
            content_type=file.content_type,
        )
        saved = True
    except HTTPException as exc:
        logger.warning(
            "upload rejected: name=%s size=%s status=%s detail=%s",
            original_name,
            size,
            exc.status_code,
            exc.detail,
        )
        raise
    except MediaValidationError as exc:
        if exc.status_code >= 500:
            logger.error(
                "media validation server error: name=%s error=%s",
                original_name,
                exc,
            )
        else:
            logger.warning(
                "media validation failed: name=%s size=%s error=%s",
                original_name,
                size,
                exc,
            )
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception:
        logger.exception("unexpected upload error: name=%s", original_name)
        raise HTTPException(status_code=500, detail="服务器处理上传文件时发生错误")
    finally:
        if not saved:
            target.unlink(missing_ok=True)

    logger.info("upload success: name=%s saved=%s size=%s", original_name, saved_name, size)
    return record


@app.post("/api/upload-local")
def upload_local_path(payload: LocalUploadRequest):
    """登记本地文件路径，不复制文件，只校验并记录元数据。"""
    media_path = Path(payload.path).expanduser().resolve()
    if not media_path.is_file():
        raise HTTPException(status_code=400, detail="文件路径不存在或不是文件")

    try:
        validate_media_file(media_path, probe=settings.ffprobe_path)
    except MediaValidationError as exc:
        if exc.status_code >= 500:
            logger.error(
                "local media validation server error: path=%s error=%s",
                media_path,
                exc,
            )
        else:
            logger.warning(
                "local media validation failed: path=%s error=%s",
                media_path,
                exc,
            )
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    stat = media_path.stat()
    try:
        record = insert_file(
            original_name=media_path.name,
            stored_path=media_path,
            source_type="local",
            size=stat.st_size,
            content_type=None,
        )
    except Exception:
        logger.exception("failed to register local path: %s", media_path)
        raise HTTPException(status_code=500, detail="登记文件信息失败")

    logger.info(
        "local path registered: name=%s path=%s size=%s",
        media_path.name,
        media_path,
        stat.st_size,
    )
    return record


@app.get("/api/files")
def list_files():
    """列出所有文件记录，网页上传和本地路径登记都会包含。"""
    return {"files": list_stored_files()}
