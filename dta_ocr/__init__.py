import sqlite3
from dta_ocr.bbox import BoundingBox


def create_progress_schema(conn: sqlite3.Connection):
    with open("schema.sql", "r") as schema:
        conn.executescript(schema.read())

    conn.commit()


__all__ = [
    "BoundingBox",
    "create_progress_schema"
]
