from summary_db import get_conn

def verify():
    try:
        conn = get_conn()
        cur = conn.cursor()
        
        # Check if table exists
        cur.execute("""
            SELECT exists (
                SELECT FROM information_schema.tables 
                WHERE  table_schema = 'public'
                AND    table_name   = 'summaries'
            );
        """)
        exists = cur.fetchone()[0]
        
        if exists:
            print("SUCCESS: Table 'summaries' exists in Neon DB.")
            
            # Check columns
            cur.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'summaries';
            """)
            columns = cur.fetchall()
            print("Columns found:")
            for col in columns:
                print(f"- {col[0]} ({col[1]})")
                
        else:
            print("FAILURE: Table 'summaries' does NOT exist.")
            
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"FAILURE: Connection error: {e}")

if __name__ == "__main__":
    verify()
