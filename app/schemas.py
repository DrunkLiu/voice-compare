"""API 请求与响应数据模型。"""

from pydantic import BaseModel, Field


class WordTimestamp(BaseModel):
    """单个单词的时间戳信息。"""

    word: str
    start: float
    end: float
    probability: float | None = None
    segment_id: int


class TranscriptionSegment(BaseModel):
    """转写结果中的一段文本。"""

    id: int
    start: float
    end: float
    text: str
    words: list[WordTimestamp] = Field(default_factory=list)


class TranscriptionResult(BaseModel):
    """转写接口的完整响应结构。"""

    text: str
    language: str
    duration: float
    segments: list[TranscriptionSegment]
    words: list[WordTimestamp]
    file_id: str | None = None
