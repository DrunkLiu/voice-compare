"""文件元数据存储：使用 SQLite 记录上传文件或本地路径，避免重复拷贝文件。"""

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.config import get_settings

settings = get_settings()
DB_PATH = settings.db_path
_db_initialized = False
_db_init_lock = threading.Lock()

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY,
    original_name TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    source_type TEXT NOT NULL,
    size INTEGER NOT NULL,
    content_type TEXT,
    created_at INTEGER NOT NULL
)
"""


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """创建数据库连接并确保正确关闭，连接默认带忙等待超时。"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row) -> dict:
    """把数据库行转换为接口可返回的字典，避免多处重复映射。"""
    return {
        "id": row["id"],
        "name": row["original_name"],
        "stored_path": row["stored_path"],
        "source_type": row["source_type"],
        "size": row["size"],
        "content_type": row["content_type"],
        "created_at": row["created_at"],
    }


def init_db() -> None:
    """初始化数据表，使用锁保证并发启动时只执行一次。"""
    global _db_initialized
    with _db_init_lock:
        if _db_initialized:
            return
        with _connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_CREATE_TABLE_SQL)
        _db_initialized = True


def _ensure_db() -> None:
    """确保数据表已初始化，供直接调用存储函数时兜底。"""
    if not _db_initialized:
        init_db()


def insert_file(
    original_name: str,
    stored_path: str | Path,
    source_type: str,
    size: int,
    content_type: str | None = None,
) -> dict:
    """写入一条文件记录，并返回可展示的元数据。"""
    _ensure_db()
    stored_path_text = str(stored_path)

    with _connect() as conn:
        existing = conn.execute(
            "SELECT * FROM files WHERE stored_path = ?",
            (stored_path_text,),
        ).fetchone()
        if existing is not None:
            conn.execute(
                "UPDATE files SET size = ?, content_type = ? WHERE id = ?",
                (size, content_type, existing["id"]),
            )
            row = conn.execute(
                "SELECT * FROM files WHERE id = ?",
                (existing["id"],),
            ).fetchone()
            return _row_to_dict(row)

        record_id = uuid4().hex
        created_at = int(datetime.now(timezone.utc).timestamp())
        conn.execute(
            """
            INSERT INTO files (
                id, original_name, stored_path, source_type, size,
                content_type, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                original_name,
                stored_path_text,
                source_type,
                size,
                content_type,
                created_at,
            ),
        )
        row = conn.execute(
            "SELECT * FROM files WHERE id = ?",
            (record_id,),
        ).fetchone()
    return _row_to_dict(row)


def list_files() -> list[dict]:
    """按创建时间倒序返回所有文件记录。"""
    _ensure_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM files ORDER BY created_at DESC, id DESC"
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def get_file(file_id: str) -> dict | None:
    """按记录 ID 查询单个文件记录，不存在时返回 None。"""
    _ensure_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM files WHERE id = ?",
            (file_id,),
        ).fetchone()

    if row is None:
        return None

    return _row_to_dict(row)
