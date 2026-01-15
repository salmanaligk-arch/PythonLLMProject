import sqlite3

def get_db():
    return sqlite3.connect("chat_history.db")

def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS chat (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT,
                response TEXT     
            )
        ''')

def insert_chat(prompt, response):
    with get_db() as conn:
        conn.execute("INSERT INTO chat (prompt, response) VALUES (?, ?)", (prompt, response))

#init_db()