import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("DB_CONNECTION_STRING")

def get_db_connection():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is not set")
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor, connect_timeout=5)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Create tables if not exist
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id UUID PRIMARY KEY,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE IF NOT EXISTS messages (
        id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
        session_id UUID REFERENCES sessions(id),
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE IF NOT EXISTS summaries (
        id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
        session_id UUID REFERENCES sessions(id),
        summary TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized.")
