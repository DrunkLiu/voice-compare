"""本地英文语音转写模块：封装 faster-whisper，模型只加载一次。"""

import logging
import os
import threading
from pathlib import Path

from faster_whisper import WhisperModel

from app.config import get_settings

logger = logging.getLogger(__name__)

# 全局缓存模型实例，避免每次请求都重新加载
_model: WhisperModel | None = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()
_MIN_WORD_DURATION = 0.01
_WORD_EDGE_CHARS = " \t\n\r.,;:!?()[]{}\"'‘’“”…—-"


def _clean_word_text(text: str) -> str:
    """去掉单词首尾标点，保留内部撇号和连字符。"""
    return text.strip().strip(_WORD_EDGE_CHARS)


def _sanitize_word_timestamps(
    words: list[dict],
    segment_start: float,
    segment_end: float,
) -> list[dict]:
    """清洗词级时间戳，保证时间单调且每个词有正时长。"""
    if not words:
        return []

    result = []
    cursor = float(segment_start)
    for index, item in enumerate(words):
        start = max(float(item["start"]), cursor)
        next_start = (
            float(words[index + 1]["start"])
            if index + 1 < len(words)
            else float(segment_end)
        )
        end = float(item["end"])
        if end <= start:
            end = next_start
        if end <= start:
            end = start + _MIN_WORD_DURATION
        result.append(
            {
                **item,
                "start": float(round(start, 3)),
                "end": float(round(end, 3)),
            }
        )
        cursor = end
    return result


def _load_model() -> WhisperModel:
    """按配置加载 Whisper 模型，不参与缓存判断。"""
    settings = get_settings()
    # 强制使用配置的 HuggingFace 镜像，解决模型下载网络问题
    os.environ["HF_ENDPOINT"] = settings.hf_endpoint
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    # 优先加载本地模型目录，避免每次启动都依赖网络
    local_model_dir = settings.whisper_model_dir / settings.whisper_model_size
    model_ref = (
        str(local_model_dir)
        if local_model_dir.is_dir()
        else settings.whisper_model_size
    )
    try:
        model = WhisperModel(
            model_ref,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            download_root=str(settings.whisper_model_dir),
            local_files_only=settings.whisper_local_files_only,
        )
    except Exception:
        logger.exception("failed to load whisper model: %s", model_ref)
        raise
    logger.info("whisper model loaded: %s", settings.whisper_model_size)
    return model


def get_model() -> WhisperModel:
    """加载并缓存 Whisper 模型，双检锁保证并发请求只加载一次。"""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = _load_model()
    return _model


def transcribe_audio(path: str | Path) -> dict:
    """转写音频文件，返回完整文本、语言、时长和分段信息。"""
    settings = get_settings()
    # 转写期间持有锁，避免多个线程同时使用同一个模型实例
    with _inference_lock:
        model = get_model()
        segments, info = model.transcribe(
            str(path),
            language=settings.whisper_language,
            beam_size=5,
            vad_filter=True,
            word_timestamps=settings.whisper_word_timestamps,
        )

        segment_list = []
        word_list = []
        for segment in segments:
            raw_words = []
            for word in getattr(segment, "words", None) or []:
                word_text = _clean_word_text(getattr(word, "word", ""))
                if not word_text:
                    continue
                probability = getattr(word, "probability", None)
                raw_words.append(
                    {
                        "word": word_text,
                        "start": float(getattr(word, "start", 0.0)),
                        "end": float(getattr(word, "end", 0.0)),
                        "probability": (
                            float(round(probability, 3))
                            if probability is not None
                            else None
                        ),
                        "segment_id": segment.id,
                    }
                )
            words = _sanitize_word_timestamps(
                raw_words,
                float(segment.start),
                float(segment.end),
            )
            word_list.extend(words)
            segment_list.append(
                {
                    "id": segment.id,
                    "start": float(round(segment.start, 3)),
                    "end": float(round(segment.end, 3)),
                    "text": segment.text.strip(),
                    "words": words,
                }
            )

    # 个别媒体信息可能缺失，这里提供默认值避免崩溃
    language = getattr(info, "language", None) or "unknown"
    duration = getattr(info, "duration", None) or 0.0

    return {
        "text": "".join(item["text"] for item in segment_list).strip(),
        "language": language,
        "duration": float(round(duration, 3)),
        "segments": segment_list,
        "words": word_list,
    }
