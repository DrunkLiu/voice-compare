"""M1 接口自动化测试：覆盖首页、健康检查、上传与文件列表。"""

import json
import subprocess
import wave
from io import BytesIO

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.media_validation import MediaValidationError, validate_media_file
from app.storage import init_db, insert_file
from app.transcriber import transcribe_audio

# TestClient 模拟真实 HTTP 请求，无需启动服务
client = TestClient(app)


@pytest.fixture(autouse=True)
def isolate_db(tmp_path, monkeypatch):
    """每个测试使用独立的临时数据库，避免污染真实 data 目录。"""
    db_path = tmp_path / "files.db"
    monkeypatch.setattr("app.storage.DB_PATH", db_path)
    monkeypatch.setattr("app.storage._db_initialized", False)
    init_db()


def make_wav_bytes() -> bytes:
    """生成一段最小但合法的 WAV 音频，用于测试真实音频上传。"""
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 800)
    return buffer.getvalue()


def test_index():
    """首页应返回上传页面，并包含 M1 标题文字。"""
    response = client.get("/")
    assert response.status_code == 200
    assert "M1" in response.text


def test_health():
    """健康检查接口应返回正常状态。"""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_and_list(tmp_path, monkeypatch):
    """上传合法音频后应能通过文件列表接口查到。"""
    # 把上传目录临时指向测试目录，避免污染真实 uploads 文件夹
    monkeypatch.setattr("app.main.UPLOAD_DIR", tmp_path)

    # 模拟一次包含文件的 POST 请求
    response = client.post(
        "/api/upload",
        files={"file": ("hello.wav", BytesIO(make_wav_bytes()), "audio/wav")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "hello.wav"
    assert data["size"] == len(make_wav_bytes())

    # 上传完成后应能在文件列表中看到该文件
    listing = client.get("/api/files")
    assert listing.status_code == 200
    files = listing.json()["files"]
    assert len(files) == 1
    assert files[0]["name"] == "hello.wav"
    assert files[0]["source_type"] == "upload"
    assert files[0]["size"] == len(make_wav_bytes())


def test_reject_image_extension(tmp_path, monkeypatch):
    """图片、动图和 Live Photo 扩展名应被拒绝。"""
    monkeypatch.setattr("app.main.UPLOAD_DIR", tmp_path)

    response = client.post(
        "/api/upload",
        files={
            "file": (
                "live.heic",
                BytesIO(b"\x00\x00\x00\x18ftypheic" + b"\x00" * 12),
                "image/heic",
            )
        },
    )
    assert response.status_code == 400
    assert "图片" in response.json()["detail"]


def test_reject_unsupported_extension(tmp_path, monkeypatch):
    """非音频/视频扩展名应返回明确错误。"""
    monkeypatch.setattr("app.main.UPLOAD_DIR", tmp_path)

    response = client.post(
        "/api/upload",
        files={"file": ("notes.txt", BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 400
    assert "音频或视频" in response.json()["detail"]


def test_reject_fake_audio(tmp_path, monkeypatch):
    """扩展名正确但内容不是音频的文件应被 ffprobe 拦截。"""
    monkeypatch.setattr("app.main.UPLOAD_DIR", tmp_path)

    response = client.post(
        "/api/upload",
        files={"file": ("fake.mp3", BytesIO(b"not really audio"), "audio/mpeg")},
    )
    assert response.status_code == 400
    assert "音频或视频" in response.json()["detail"]


def test_validate_media_file_accepts_wav(tmp_path):
    """独立校验函数应接受合法 WAV 文件。"""
    wav_path = tmp_path / "ok.wav"
    wav_path.write_bytes(make_wav_bytes())
    validate_media_file(wav_path)


def test_validate_media_file_rejects_image(tmp_path):
    """独立校验函数应拒绝图片/动图文件。"""
    image_path = tmp_path / "fake.gif"
    image_path.write_bytes(b"GIF89a\x01\x00\x01\x00\x00\x00\x00;")
    with pytest.raises(MediaValidationError):
        validate_media_file(image_path)


def test_reject_empty_file(tmp_path, monkeypatch):
    """空文件应返回明确错误。"""
    monkeypatch.setattr("app.main.UPLOAD_DIR", tmp_path)

    response = client.post(
        "/api/upload",
        files={"file": ("empty.wav", BytesIO(b""), "audio/wav")},
    )
    assert response.status_code == 400
    assert "空" in response.json()["detail"]


def test_reject_oversize_file(tmp_path, monkeypatch):
    """超过大小限制的文件应返回 413。"""
    monkeypatch.setattr("app.main.UPLOAD_DIR", tmp_path)
    monkeypatch.setattr("app.main.MAX_UPLOAD_BYTES", 10)

    response = client.post(
        "/api/upload",
        files={"file": ("big.wav", BytesIO(make_wav_bytes()), "audio/wav")},
    )
    assert response.status_code == 413


def test_ffprobe_missing_returns_500(tmp_path, monkeypatch):
    """ffprobe 缺失属于服务端错误，应返回 500。"""
    monkeypatch.setattr("app.main.UPLOAD_DIR", tmp_path)
    monkeypatch.setattr("app.media_validation.shutil.which", lambda _: None)

    response = client.post(
        "/api/upload",
        files={"file": ("ok.wav", BytesIO(make_wav_bytes()), "audio/wav")},
    )
    assert response.status_code == 500
    assert "ffprobe" in response.json()["detail"]


def test_upload_local_path_records_without_copy(tmp_path, monkeypatch):
    """本地路径登记应只记录路径，不把文件复制到 uploads。"""
    source = tmp_path / "source.wav"
    source.write_bytes(make_wav_bytes())
    uploads = tmp_path / "uploads"
    monkeypatch.setattr("app.main.UPLOAD_DIR", uploads)

    response = client.post("/api/upload-local", json={"path": str(source)})
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "source.wav"
    assert data["source_type"] == "local"
    assert data["stored_path"] == str(source.resolve())

    # 不应产生任何副本
    assert not uploads.exists() or not list(uploads.iterdir())

    listing = client.get("/api/files")
    files = listing.json()["files"]
    assert files[0]["stored_path"] == str(source.resolve())


def test_upload_local_path_missing_returns_400(tmp_path, monkeypatch):
    """不存在的本地路径应返回 400。"""
    monkeypatch.setattr("app.main.UPLOAD_DIR", tmp_path / "uploads")

    response = client.post(
        "/api/upload-local",
        json={"path": str(tmp_path / "missing.wav")},
    )
    assert response.status_code == 400
    assert "路径" in response.json()["detail"]


def test_transcribe_record_success(tmp_path, monkeypatch):
    """转写接口应返回文本和分段信息。"""
    source = tmp_path / "demo.wav"
    source.write_bytes(make_wav_bytes())
    record = insert_file(
        original_name="demo.wav",
        stored_path=source,
        source_type="local",
        size=len(make_wav_bytes()),
    )

    fake_result = {
        "text": "hello world",
        "language": "en",
        "duration": 1.0,
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.0,
                "text": "hello world",
                "words": [
                    {
                        "word": "hello",
                        "start": 0.0,
                        "end": 0.4,
                        "probability": 0.98,
                        "segment_id": 0,
                    }
                ],
            },
        ],
        "words": [
            {
                "word": "hello",
                "start": 0.0,
                "end": 0.4,
                "probability": 0.98,
                "segment_id": 0,
            }
        ],
    }
    monkeypatch.setattr("app.main.transcribe_audio", lambda path: fake_result)

    response = client.post("/api/transcribe", json={"file_id": record["id"]})
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "hello world"
    assert data["file_id"] == record["id"]
    assert data["segments"][0]["text"] == "hello world"
    assert data["words"][0]["word"] == "hello"


def test_transcribe_record_not_found(tmp_path, monkeypatch):
    """不存在的文件记录应返回 404。"""
    monkeypatch.setattr("app.main.UPLOAD_DIR", tmp_path / "uploads")

    response = client.post("/api/transcribe", json={"file_id": "not-exist"})
    assert response.status_code == 404


def test_transcribe_missing_file_returns_400(tmp_path, monkeypatch):
    """记录存在但文件已丢失时应返回 400。"""
    missing_path = tmp_path / "missing.wav"
    record = insert_file(
        original_name="missing.wav",
        stored_path=missing_path,
        source_type="local",
        size=0,
    )

    response = client.post("/api/transcribe", json={"file_id": record["id"]})
    assert response.status_code == 400
    assert "文件" in response.json()["detail"]


def test_transcribe_audio_builds_result(monkeypatch):
    """转写核心模块应把生成器结果转换为统一结构。"""

    class FakeWord:
        word = " hello "
        start = 0.0
        end = 0.5
        probability = 0.99

    class FakeSegment:
        def __init__(self):
            self.id = 0
            self.start = 0.0
            self.end = 1.0
            self.text = " hello "
            self.words = [FakeWord()]

    class FakeInfo:
        language = "en"
        duration = 1.0

    class FakeModel:
        def transcribe(self, path, **kwargs):
            return iter([FakeSegment()]), FakeInfo()

    monkeypatch.setattr("app.transcriber.get_model", lambda: FakeModel())

    result = transcribe_audio("demo.wav")

    assert result == {
        "text": "hello",
        "language": "en",
        "duration": 1.0,
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.0,
                "text": "hello",
                "words": [
                    {
                        "word": "hello",
                        "start": 0.0,
                        "end": 0.5,
                        "probability": 0.99,
                        "segment_id": 0,
                    }
                ],
            }
        ],
        "words": [
            {
                "word": "hello",
                "start": 0.0,
                "end": 0.5,
                "probability": 0.99,
                "segment_id": 0,
            }
        ],
    }


def test_transcribe_audio_passes_word_timestamps(monkeypatch):
    """转写时应按配置开启词级时间戳。"""
    captured = {}

    class FakeInfo:
        pass

    class FakeModel:
        def transcribe(self, path, **kwargs):
            captured.update(kwargs)
            return iter([]), FakeInfo()

    monkeypatch.setattr("app.transcriber.get_model", lambda: FakeModel())

    transcribe_audio("demo.wav")

    assert captured["word_timestamps"] is True


def test_transcribe_audio_uses_defaults_for_missing_info(monkeypatch):
    """媒体信息缺失时，转写模块应使用默认值而不是崩溃。"""

    class FakeInfo:
        pass

    class FakeModel:
        def transcribe(self, path, **kwargs):
            return iter([]), FakeInfo()

    monkeypatch.setattr("app.transcriber.get_model", lambda: FakeModel())

    result = transcribe_audio("demo.wav")

    assert result["text"] == ""
    assert result["language"] == "unknown"
    assert result["duration"] == 0.0
    assert result["segments"] == []
    assert result["words"] == []


def test_validate_media_file_requires_audio_stream(tmp_path, monkeypatch):
    """无声视频即使格式合法，也应因缺少音频流被拒绝。"""
    media_path = tmp_path / "silent.mp4"
    media_path.write_bytes(b"fake")
    monkeypatch.setattr("app.media_validation.filetype.is_image", lambda _: False)
    fake_result = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=json.dumps({"streams": [{"codec_type": "video"}]}),
        stderr="",
    )
    monkeypatch.setattr(
        "app.media_validation.subprocess.run",
        lambda *args, **kwargs: fake_result,
    )

    with pytest.raises(MediaValidationError, match="音频"):
        validate_media_file(media_path)


def test_upload_local_path_rejects_oversize(tmp_path, monkeypatch):
    """本地路径登记应复用上传的大小限制。"""
    source = tmp_path / "big.wav"
    source.write_bytes(make_wav_bytes())
    monkeypatch.setattr("app.main.UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr("app.main.MAX_UPLOAD_BYTES", 10)

    response = client.post("/api/upload-local", json={"path": str(source)})

    assert response.status_code == 413


def test_upload_local_path_deduplicates(tmp_path, monkeypatch):
    """同一路径重复登记应复用已有记录，而不是产生重复文件列表项。"""
    source = tmp_path / "demo.wav"
    source.write_bytes(make_wav_bytes())
    monkeypatch.setattr("app.main.UPLOAD_DIR", tmp_path / "uploads")

    first = client.post("/api/upload-local", json={"path": str(source)})
    second = client.post("/api/upload-local", json={"path": str(source)})

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert len(client.get("/api/files").json()["files"]) == 1


def test_unhandled_exception_returns_500(monkeypatch):
    """未捕获异常应被全局处理器捕获，返回 500 而不是让服务崩溃。"""

    def raise_error():
        raise RuntimeError("boom")

    monkeypatch.setattr("app.main.list_stored_files", raise_error)
    test_client = TestClient(app, raise_server_exceptions=False)
    response = test_client.get("/api/files")
    assert response.status_code == 500
    assert "内部" in response.json()["detail"]


def test_lifespan_creates_directories(tmp_path, monkeypatch):
    """应用启动时应自动创建上传目录和静态目录。"""
    upload_dir = tmp_path / "uploads"
    static_dir = tmp_path / "static"
    monkeypatch.setattr("app.main.UPLOAD_DIR", upload_dir)
    monkeypatch.setattr("app.main.STATIC_DIR", static_dir)

    with TestClient(app):
        assert upload_dir.is_dir()
        assert static_dir.is_dir()
