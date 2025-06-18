import sys
import os

# backend 디렉토리를 Python 경로에 추가
backend_path = '/workspace/open-webui/backend'
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

def create_usage_tables():
    """사용량 추적 테이블을 수동으로 생성"""
    try:
        from open_webui.internal.db import engine
        from sqlalchemy import text
        
        print("Creating usage tracking tables...")
        
        # UserAPIUsage 테이블 생성 SQL
        create_usage_table = """
        CREATE TABLE IF NOT EXISTS user_api_usage (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            api_type TEXT NOT NULL,
            url_idx INTEGER NOT NULL,
            daily_tokens INTEGER DEFAULT 0,
            monthly_tokens INTEGER DEFAULT 0,
            daily_requests INTEGER DEFAULT 0,
            monthly_requests INTEGER DEFAULT 0,
            daily_cost REAL DEFAULT 0.0,
            monthly_cost REAL DEFAULT 0.0,
            last_daily_reset TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_monthly_reset TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        # UserAPILimit 테이블 생성 SQL
        create_limit_table = """
        CREATE TABLE IF NOT EXISTS user_api_limits (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL UNIQUE,
            daily_token_limit INTEGER DEFAULT 50000,
            monthly_token_limit INTEGER DEFAULT 500000,
            daily_request_limit INTEGER DEFAULT 200,
            monthly_request_limit INTEGER DEFAULT 2000,
            daily_cost_limit REAL DEFAULT 10.0,
            monthly_cost_limit REAL DEFAULT 100.0,
            enabled BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        # 인덱스 생성 SQL
        create_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_user_api_usage_user_id ON user_api_usage(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_user_api_usage_api_type ON user_api_usage(api_type)",
            "CREATE INDEX IF NOT EXISTS idx_user_api_limits_user_id ON user_api_limits(user_id)"
        ]
        
        # 테이블 생성 실행
        with engine.connect() as conn:
            conn.execute(text(create_usage_table))
            print("***user_api_usage 테이블이 생성되었습니다.")
            
            conn.execute(text(create_limit_table))
            print("***user_api_limits 테이블이 생성되었습니다.")
            
            for idx_sql in create_indexes:
                conn.execute(text(idx_sql))
            print("***인덱스가 생성되었습니다.")
            
            conn.commit()
        
        # 테이블 확인
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%usage%'"))
            tables = result.fetchall()
            print(f"***생성된 테이블: {[table[0] for table in tables]}")
        
        print("**사용량 테이블이 생성되었습니다.")
        
    except Exception as e:
        print(f"**에러: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_usage_tables()