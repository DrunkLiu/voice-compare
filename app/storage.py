"""文件元数据存储：使用 SQLite 记录上传文件或本地路径，避免重复拷贝文件。"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.config import get_settings

settings = get_settings()
DB_PATH = settings.db_path

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


def _connect() -> sqlite3.Connection:
    """创建数据库连接，并确保父目录存在。"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """初始化数据表，重复调用是安全的。"""
    conn = _connect()
    try:
        conn.execute(_CREATE_TABLE_SQL)
        conn.commit()
    finally:
        conn.close()


def insert_file(
    original_name: str,
    stored_path: str | Path,
    source_type: str,
    size: int,
    content_type: str | None = None,
) -> dict:
    """写入一条文件记录，并返回可展示的元数据。"""
    init_db()
    record_id = uuid4().hex
    created_at = int(datetime.now(timezone.utc).timestamp())
    stored_path_text = str(stored_path)

    conn = _connect()
    try:
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
        conn.commit()
    finally:
        conn.close()

    return {
        "id": record_id,
        "name": original_name,
        "stored_path": stored_path_text,
        "source_type": source_type,
        "size": size,
        "content_type": content_type,
        "created_at": created_at,
    }


def list_files() -> list[dict]:
    """按创建时间倒序返回所有文件记录。"""
    init_db()
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM files ORDER BY created_at DESC, id DESC"
        ).fetchall()
    finally:
        conn.close()

    return [
        {
            "id": row["id"],
            "name": row["original_name"],
            "stored_path": row["stored_path"],
            "source_type": row["source_type"],
            "size": row["size"],
            "content_type": row["content_type"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]
