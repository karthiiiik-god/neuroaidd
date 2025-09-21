import sqlite3

DB_PATH = "neuroaid.db"

def init_db():
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS moods(
            ts TEXT, audio REAL, face REAL, fused REAL, emoji TEXT, note TEXT
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS interventions(
            ts TEXT, before REAL, after REAL, delta REAL, label TEXT
        )""")

def save_mood(audio, face, fused, emoji, note=""):
    with sqlite3.connect(DB_PATH) as c:
        c.execute("INSERT INTO moods VALUES(datetime('now'),?,?,?,?,?)",
                  (audio, face, fused, emoji, note))
