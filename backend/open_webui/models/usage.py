import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, create_engine, UniqueConstraint, Index
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert
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
    api_type = Column(String, nullable=False, index=True)
    url_idx = Column(Integer, nullable=False)

    daily_tokens = Column(Integer, default=0)
    monthly_tokens = Column(Integer, default=0)
    yearly_tokens = Column(Integer, default=0)
    daily_requests = Column(Integer, default=0)
    monthly_requests = Column(Integer, default=0)
    yearly_requests = Column(Integer, default=0)

    daily_cost = Column(Float, default=0.0)
    monthly_cost = Column(Float, default=0.0)
    yearly_cost = Column(Float, default=0.0)

    last_daily_reset = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_monthly_reset = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_yearly_reset = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index('idx_user_api_usage_unique', 'user_id', 'api_type', 'url_idx', unique=True),
    )

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

    @staticmethod
    def upsert_user_usage(user_id: str, api_type: str, url_idx: int, **update_values) -> UserAPIUsage:
        """
        사용자 사용량을 upsert (INSERT ... ON CONFLICT UPDATE)
        Race condition 방지를 위해 PostgreSQL의 ON CONFLICT 사용
        """
        from open_webui.internal.db import get_db, engine

        # PostgreSQL 여부 확인
        is_postgresql = 'postgresql' in str(engine.url)

        if is_postgresql:
            # PostgreSQL: ON CONFLICT 사용
            from sqlalchemy.dialects.postgresql import insert as pg_insert

            with get_db() as db:
                now_utc = datetime.now(timezone.utc)

                # INSERT 값 준비
                insert_values = {
                    'id': str(uuid.uuid4()),
                    'user_id': user_id,
                    'api_type': api_type,
                    'url_idx': url_idx,
                    'daily_tokens': 0,
                    'monthly_tokens': 0,
                    'yearly_tokens': 0,
                    'daily_requests': 0,
                    'monthly_requests': 0,
                    'yearly_requests': 0,
                    'daily_cost': 0.0,
                    'monthly_cost': 0.0,
                    'yearly_cost': 0.0,
                    'last_daily_reset': now_utc,
                    'last_monthly_reset': now_utc,
                    'last_yearly_reset': now_utc,
                    'created_at': now_utc,
                    'updated_at': now_utc,
                }

                # ON CONFLICT UPDATE 값 준비
                update_dict = {**update_values, 'updated_at': now_utc}

                stmt = pg_insert(UserAPIUsage).values(**insert_values)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['user_id', 'api_type', 'url_idx'],
                    set_=update_dict
                ).returning(UserAPIUsage)

                result = db.execute(stmt)
                db.commit()

                # 결과 조회
                return db.query(UserAPIUsage).filter(
                    UserAPIUsage.user_id == user_id,
                    UserAPIUsage.api_type == api_type,
                    UserAPIUsage.url_idx == url_idx
                ).first()
        else:
            # SQLite: 기존 방식 (get-or-create + update)
            with get_db() as db:
                usage = db.query(UserAPIUsage).filter(
                    UserAPIUsage.user_id == user_id,
                    UserAPIUsage.api_type == api_type,
                    UserAPIUsage.url_idx == url_idx
                ).first()

                if not usage:
                    usage = UserAPIUsage(
                        user_id=user_id,
                        api_type=api_type,
                        url_idx=url_idx
                    )
                    db.add(usage)

                for key, value in update_values.items():
                    if hasattr(usage, key):
                        setattr(usage, key, value)

                db.commit()
                db.refresh(usage)
                return usage

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