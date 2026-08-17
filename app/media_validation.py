"""媒体文件校验工具：复用 filetype 与 ffprobe，判断上传文件是否为可用的音频或视频。"""

import json
import shutil
import subprocess
from pathlib import Path

import filetype


class MediaValidationError(ValueError):
    """媒体文件校验失败时抛出的异常，status_code 区分客户端/服务端错误。"""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def is_image_bytes(data: bytes) -> bool:
    """按文件头判断字节内容是否为图片/动图。"""
    detected = filetype.guess(data)
    return detected is not None and detected.mime.startswith("image/")


def validate_media_file(path: str | Path, probe: str | None = None) -> None:
    """校验文件内容是否为音频或视频，失败时抛出 MediaValidationError。"""
    media_path = Path(path)

    # 第一层：filetype 按文件头识别真实类型，拦截图片/动图/Live Photo
    if filetype.is_image(str(media_path)):
        raise MediaValidationError("不支持图片、动图或 Live Photo，请上传音频或视频")

    # 第二层：ffprobe 确认文件包含音频流或视频流
    probe = probe or shutil.which("ffprobe")
    if not probe:
        raise MediaValidationError("服务器缺少 ffprobe，无法校验文件", status_code=500)

    try:
        result = subprocess.run(
            [
                probe,
                "-v",
                "error",
                "-show_entries",
                "format=format_name:stream=codec_type",
                "-of",
                "json",
                str(media_path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaValidationError(
            "文件校验超时，请稍后重试",
            status_code=500,
        ) from exc

    if result.returncode != 0:
        raise MediaValidationError("文件无法被识别为有效的音频或视频")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaValidationError("文件无法被识别为有效的音频或视频") from exc

    # 发音对比必须依赖音频，纯视频流（无声视频）在这里直接拦截
    streams = data.get("streams", [])
    if not any(stream.get("codec_type") == "audio" for stream in streams):
        raise MediaValidationError("文件中没有可用的音频流，请上传带声音的音频或视频")
