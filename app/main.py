"""FastAPI 后端入口：提供页面、健康检查、上传和文件列表接口。"""

import logging
import re
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import get_settings
from app.media_validation import (
    MediaValidationError,
    is_image_bytes,
    validate_media_file,
)
from app.schemas import TranscriptionResult
from app.storage import get_file, init_db, insert_file
from app.storage import list_files as list_stored_files
from app.transcriber import transcribe_audio

settings = get_settings()


def _setup_logging() -> None:
    """配置控制台日志和滚动文件日志，文件最大 5MB，保留 3 份。"""
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handlers = [logging.StreamHandler()]
    try:
        file_handler = RotatingFileHandler(
            log_dir / "app.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        handlers.append(file_handler)
    except OSError:
        # 文件日志失败时仍保留控制台日志，不影响服务启动
        print("WARNING: failed to create file logger")

    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=settings.log_level, handlers=handlers)


_setup_logging()
logger = logging.getLogger(__name__)

UPLOAD_DIR = settings.upload_dir
STATIC_DIR = settings.static_dir
MAX_UPLOAD_BYTES = settings.max_upload_size_mb * 1024 * 1024
CHUNK_SIZE = 1024 * 1024


class LocalUploadRequest(BaseModel):
    """本地路径登记请求体。"""

    path: str


class TranscribeRequest(BaseModel):
    """转写请求体。"""

    file_id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时创建必要目录。"""
    try:
        init_db()
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        STATIC_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.exception("failed to create app directories")
        raise
    yield


app = FastAPI(title=settings.app_name, version=settings.version, lifespan=lifespan)
# 把 /static 路径映射到 static 目录，供浏览器加载静态资源
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """兜底捕获所有未处理异常，记录日志并返回统一 500 响应。"""
    logger.exception(
        "unhandled exception: method=%s path=%s",
        request.method,
        request.url.path,
    )
    return JSONResponse(status_code=500, content={"detail": "服务器内部错误"})


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


def _reject_oversize(size: int) -> None:
    """统一的大小限制检查，上传和本地路径登记共用。"""
    if size > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"文件超过大小限制（{settings.max_upload_size_mb}MB）",
        )


def _handle_media_validation_error(exc: MediaValidationError, context: str) -> None:
    """统一处理媒体校验异常，按服务端/客户端错误区分日志级别。"""
    if exc.status_code >= 500:
        logger.error("media validation server error: %s error=%s", context, exc)
    else:
        logger.warning("media validation failed: %s error=%s", context, exc)
    raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.get("/", include_in_schema=False)
def index():
    """根路径返回上传页面。"""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    """健康检查接口，用于确认服务是否正常运行。"""
    return {"status": "ok", "version": settings.version}


@app.post("/api/upload")
def upload_file(request: Request, file: UploadFile = File(...)):  # noqa: B008
    """接收上传文件，校验格式和大小后保存到 uploads 目录。"""
    original_name = Path(file.filename or "unknown").name
    saved_name = _safe_saved_name(original_name)
    target = UPLOAD_DIR / saved_name
    size = 0
    saved = False

    # 根据 Content-Length 提前拒绝超大文件，避免浪费带宽
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit():
        _reject_oversize(int(content_length))

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
        _reject_oversize(size)

        # 分块写入磁盘，避免大文件一次性读入内存
        with target.open("wb") as buffer:
            buffer.write(first_chunk)
            while chunk := file.file.read(CHUNK_SIZE):
                size += len(chunk)
                _reject_oversize(size)
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
        _handle_media_validation_error(
            exc,
            f"name={original_name} size={size}",
        )
    except Exception:
        logger.exception("unexpected upload error: name=%s", original_name)
        raise HTTPException(status_code=500, detail="服务器处理上传文件时发生错误")
    finally:
        if not saved:
            target.unlink(missing_ok=True)

    logger.info(
        "upload success: name=%s saved=%s size=%s", original_name, saved_name, size
    )
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
        _handle_media_validation_error(exc, f"path={media_path}")

    stat = media_path.stat()
    _reject_oversize(stat.st_size)
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


@app.post("/api/transcribe", response_model=TranscriptionResult)
def transcribe_record(payload: TranscribeRequest):
    """根据文件记录 ID 转写音频，返回文本和分段信息。"""
    record = get_file(payload.file_id)
    if record is None:
        raise HTTPException(status_code=404, detail="文件记录不存在")

    media_path = Path(record["stored_path"])
    if not media_path.is_file():
        raise HTTPException(status_code=400, detail="文件路径已失效或文件不存在")

    try:
        result = transcribe_audio(media_path)
    except Exception:
        logger.exception(
            "transcription failed: id=%s path=%s",
            payload.file_id,
            media_path,
        )
        raise HTTPException(status_code=500, detail="转写失败，请检查模型或音频文件")

    result["file_id"] = payload.file_id
    logger.info(
        "transcription success: id=%s duration=%s",
        payload.file_id,
        result["duration"],
    )
    return result


@app.get("/api/files")
def list_files():
    """列出所有文件记录，网页上传和本地路径登记都会包含。"""
    return {"files": list_stored_files()}
