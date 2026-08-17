"""M1 接口自动化测试：覆盖首页、健康检查、上传与文件列表。"""

from io import BytesIO
import wave

import pytest
from fastapi.testclient import TestClient

from app.media_validation import MediaValidationError, validate_media_file
from app.main import app

# TestClient 模拟真实 HTTP 请求，无需启动服务
client = TestClient(app)


@pytest.fixture(autouse=True)
def isolate_db(tmp_path, monkeypatch):
    """每个测试使用独立的临时数据库，避免污染真实 data 目录。"""
    monkeypatch.setattr("app.storage.DB_PATH", tmp_path / "files.db")


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
