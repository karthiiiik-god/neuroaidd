# src/db/store.py
import os
import sqlite3
from typing import List, Tuple, Dict, Any

DB_PATH = os.environ.get("NEUROAID_DB", "neuroaid.db")


# --------- low-level helpers ---------
def _conn():
    # check_same_thread=False lets Streamlit reuse this connection
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def _table_has_column(table: str, column: str) -> bool:
    con = _conn()
    try:
        cur = con.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        cols = [row[1] for row in cur.fetchall()]
        return column in cols
    finally:
        con.close()


# --------- schema management ---------
def init_db() -> None:
    """Create base tables if they don't exist."""
    con = _conn()
    try:
        cur = con.cursor()

        # users
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT,
                created_ts DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # moods
        cur.execute("""
            CREATE TABLE IF NOT EXISTS moods (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER NOT NULL,
                audio REAL,
                face REAL,
                fused REAL,
                emoji TEXT,
                note TEXT,
                emotion TEXT,
                tips_used TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)

        con.commit()
    finally:
        con.close()


def migrate_db() -> None:
    """
    Add new columns safely if older DBs are missing them.
    Call this at startup after init_db().
    """
    con = _conn()
    try:
        cur = con.cursor()

        # Example: ensure tips_used exists
        if not _table_has_column("moods", "tips_used"):
            cur.execute("ALTER TABLE moods ADD COLUMN tips_used TEXT")

        # You can add more migrations similarly:
        # if not _table_has_column("moods", "some_new_col"):
        #     cur.execute("ALTER TABLE moods ADD COLUMN some_new_col TEXT")

        con.commit()
    finally:
        con.close()


# --------- user helpers ---------
def get_or_create_user(username: str) -> Dict[str, Any]:
    username = (username or "").strip()
    if not username:
        raise ValueError("username is required")

    init_db()          # ensure tables exist
    con = _conn()
    try:
        cur = con.cursor()
        cur.execute("SELECT id, username FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        if row:
            return {"id": int(row[0]), "username": row[1]}

        cur.execute("INSERT INTO users (username) VALUES (?)", (username,))
        con.commit()
        return {"id": int(cur.lastrowid), "username": username}
    finally:
        con.close()


# --------- mood I/O ---------
def save_mood(
    user_id: int,
    audio: float,
    face: float,
    fused: float,
    emoji: str,
    note: str,
    emotion: str,
    tips_used: Any,   # list or str
) -> None:
    """
    Insert one mood row. tips_used can be a Python list or a string;
    we'll store text (CSV or JSON-ish).
    """
    if isinstance(tips_used, (list, tuple)):
        tips_text = ", ".join(map(str, tips_used))
    else:
        tips_text = str(tips_used) if tips_used is not None else ""

    con = _conn()
    try:
        cur = con.cursor()
        cur.execute("""
            INSERT INTO moods (user_id, audio, face, fused, emoji, note, emotion, tips_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (int(user_id), float(audio or 0.0), float(face or 0.0), float(fused or 0.0),
              emoji or "", note or "", emotion or "", tips_text))
        con.commit()
    finally:
        con.close()


def export_all(user_id: int) -> List[Tuple]:
    """
    Return rows ordered by ts ascending with the exact columns app.py expects:
    ["ts","user_id","audio","face","fused","emoji","note","emotion","tips_used"]
    """
    con = _conn()
    try:
        cur = con.cursor()
        cur.execute("""
            SELECT ts, user_id, audio, face, fused, emoji, note, emotion, tips_used
            FROM moods
            WHERE user_id = ?
            ORDER BY ts ASC
        """, (int(user_id),))
        return cur.fetchall()
    finally:
        con.close()


def load_moods(days: int = 30) -> List[Dict[str, Any]]:
    """
    Convenience loader for the last N days across all users.
    Returns a list of dicts. (Kept for backward-compat.)
    """
    con = _conn()
    try:
        cur = con.cursor()
        cur.execute(f"""
            SELECT ts, user_id, audio, face, fused, emoji, note, emotion, tips_used
            FROM moods
            WHERE ts >= datetime('now', ?)
            ORDER BY ts ASC
        """, (f'-{int(days)} days',))
        rows = cur.fetchall()
        cols = ["ts","user_id","audio","face","fused","emoji","note","emotion","tips_used"]
        return [dict(zip(cols, r)) for r in rows]
    finally:
        con.close()
