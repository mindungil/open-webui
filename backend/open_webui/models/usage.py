import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, create_engine
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel

# 순환 import 방지를 위해 여기서 Base를 다시 정의하거나 지연 import 사용
try:
    from open_webui.internal.db import Base
except ImportError:
    # Base가 없으면 새로 생성
    from sqlalchemy.ext.declarative import declarative_base
    Base = declarative_base()

####################
# 데이터베이스 모델
####################

class UserAPIUsage(Base):
    __tablename__ = "user_api_usage"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    api_type = Column(String, nullable=False)
    url_idx = Column(Integer, nullable=False)
    
    daily_tokens = Column(Integer, default=0)
    monthly_tokens = Column(Integer, default=0)
    daily_requests = Column(Integer, default=0)
    monthly_requests = Column(Integer, default=0)
    
    daily_cost = Column(Float, default=0.0)
    monthly_cost = Column(Float, default=0.0)
    
    last_daily_reset = Column(DateTime, default=func.now())
    last_monthly_reset = Column(DateTime, default=func.now())
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class UserAPILimit(Base):
    __tablename__ = "user_api_limits"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, unique=True, index=True)
    
    daily_token_limit = Column(Integer, default=50000)
    monthly_token_limit = Column(Integer, default=500000)
    daily_request_limit = Column(Integer, default=200)
    monthly_request_limit = Column(Integer, default=2000)
    
    daily_cost_limit = Column(Float, default=10.0)
    monthly_cost_limit = Column(Float, default=100.0)
    
    enabled = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

# 나머지 Pydantic 모델들과 테이블 접근 함수들은 지연 import 사용
class UserAPIUsageTable:
    
    @staticmethod
    def get_user_usage(user_id: str, api_type: str, url_idx: int) -> Optional[UserAPIUsage]:
        """사용자의 특정 API 사용량 조회"""
        from open_webui.internal.db import get_db
        
        with get_db() as db:
            return db.query(UserAPIUsage).filter(
                UserAPIUsage.user_id == user_id,
                UserAPIUsage.api_type == api_type,
                UserAPIUsage.url_idx == url_idx
            ).first()
    
    @staticmethod
    def create_user_usage(user_id: str, api_type: str, url_idx: int) -> UserAPIUsage:
        """새로운 사용량 레코드 생성"""
        from open_webui.internal.db import get_db
        
        with get_db() as db:
            usage = UserAPIUsage(
                user_id=user_id,
                api_type=api_type,
                url_idx=url_idx
            )
            db.add(usage)
            db.commit()
            db.refresh(usage)
            return usage
    
    @staticmethod
    def update_user_usage(usage_id: str, **kwargs) -> Optional[UserAPIUsage]:
        """사용량 레코드 업데이트"""
        from open_webui.internal.db import get_db
        
        with get_db() as db:
            usage = db.query(UserAPIUsage).filter(UserAPIUsage.id == usage_id).first()
            if usage:
                for key, value in kwargs.items():
                    if hasattr(usage, key):
                        setattr(usage, key, value)
                db.commit()
                db.refresh(usage)
            return usage
    
    @staticmethod
    def get_all_user_usage(user_id: str) -> list:
        """사용자의 모든 API 사용량 조회"""
        from open_webui.internal.db import get_db
        
        with get_db() as db:
            return db.query(UserAPIUsage).filter(UserAPIUsage.user_id == user_id).all()

class UserAPILimitTable:
    
    @staticmethod
    def get_user_limits(user_id: str) -> Optional[UserAPILimit]:
        """사용자의 API 제한 설정 조회"""
        from open_webui.internal.db import get_db
        
        with get_db() as db:
            return db.query(UserAPILimit).filter(UserAPILimit.user_id == user_id).first()
    
    @staticmethod
    def create_user_limits(user_id: str, **kwargs) -> UserAPILimit:
        """새로운 제한 설정 생성"""
        from open_webui.internal.db import get_db
        
        with get_db() as db:
            limits = UserAPILimit(user_id=user_id, **kwargs)
            db.add(limits)
            db.commit()
            db.refresh(limits)
            return limits
    
    @staticmethod
    def update_user_limits(user_id: str, **kwargs) -> Optional[UserAPILimit]:
        """제한 설정 업데이트"""
        from open_webui.internal.db import get_db
        
        with get_db() as db:
            limits = db.query(UserAPILimit).filter(UserAPILimit.user_id == user_id).first()
            if limits:
                for key, value in kwargs.items():
                    if hasattr(limits, key):
                        setattr(limits, key, value)
                db.commit()
                db.refresh(limits)
            else:
                limits = UserAPILimitTable.create_user_limits(user_id, **kwargs)
            return limits