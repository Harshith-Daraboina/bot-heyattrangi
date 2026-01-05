import os
import psycopg2
from datetime import datetime, timezone
import json
from dotenv import load_dotenv

load_dotenv()

def get_conn():
    conn_str = os.getenv("DB_CONNECTION_STRING")
    if not conn_str:
        raise ValueError("DB_CONNECTION_STRING not found in environment variables.")
    return psycopg2.connect(conn_str)

def init_db():
    try:
        conn = get_conn()
        cur = conn.cursor()

        # Iterate over DDL commands to ensure clarity and correct execution
        commands = [
            """
            CREATE TABLE IF NOT EXISTS summaries (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                conversation TEXT,
                summary TEXT
            )
            """
        ]

        for command in commands:
            cur.execute(command)

        conn.commit()
        cur.close()
        conn.close()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Error initializing database: {e}")
        # Not raising here to prevent app crash on startup if DB is down, 
        # but user will see error log.
        pass

def save_summary(conversation, summary_text):
    try:
        conn = get_conn()
        cur = conn.cursor()

        cur.execute(
            "INSERT INTO summaries (created_at, conversation, summary) VALUES (%s, %s, %s)",
            (
                datetime.now(timezone.utc),
                json.dumps(conversation),
                summary_text
            )
        )

        conn.commit()
        cur.close()
        conn.close()
        print("Summary saved to database.")
    except Exception as e:
        print(f"Error saving summary: {e}")
        raise e
