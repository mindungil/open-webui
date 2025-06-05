import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from open_webui.models.usage import UserAPIUsageTable, UserAPILimitTable

log = logging.getLogger(__name__)

class APIUsageTracker:
    """API 사용량 추적 및 제한 관리 클래스"""
    
    @staticmethod
    def get_external_usage_sum(user_id):
        """
        외부 API(vllm:8000 제외, 즉 url_idx > 0) 전체 사용량 합산
        """
        usages = UserAPIUsageTable.get_all_user_usage(user_id)
        total = {
            "daily_tokens": 0,
            "monthly_tokens": 0,
            "daily_requests": 0,
            "monthly_requests": 0,
            "daily_cost": 0.0,
            "monthly_cost": 0.0,
        }
        for usage in usages:
            if usage.url_idx > 0:  # vllm:8000이 아니면
                total["daily_tokens"] += usage.daily_tokens
                total["monthly_tokens"] += usage.monthly_tokens
                total["daily_requests"] += usage.daily_requests
                total["monthly_requests"] += usage.monthly_requests
                total["daily_cost"] += usage.daily_cost
                total["monthly_cost"] += usage.monthly_cost
        return total

    @staticmethod
    async def check_user_limits(user: Any, url_idx: int, estimated_tokens: int = 0) -> Dict[str, Any]:
        """사용자의 API 사용 제한을 확인"""
        try:
            # 로컬(vllm:8000)은 제한 없음
            if not APIUsageTracker.is_external_api(url_idx):
                return {"allowed": True, "reason": "local_api"}
            
            # 관리자는 제한 없음
            if hasattr(user, 'role') and user.role == 'admin':
                return {"allowed": True, "reason": "admin_user"}
            
            # 사용자별 제한 설정 조회
            limits = UserAPILimitTable.get_user_limits(user.id)
            if not limits or not limits.enabled:
                # 제한이 설정되지 않았거나 비활성화된 경우 기본 허용
                return {"allowed": True, "reason": "no_limits_set"}
            
            # 외부 API 전체 사용량 합산
            usage_sum = APIUsageTracker.get_external_usage_sum(user.id)
            
            checks = [
                {
                    "type": "daily_tokens",
                    "current": usage_sum["daily_tokens"],
                    "limit": limits.daily_token_limit,
                    "estimated": estimated_tokens,
                    "unit": "tokens"
                },
                {
                    "type": "monthly_tokens", 
                    "current": usage_sum["monthly_tokens"],
                    "limit": limits.monthly_token_limit,
                    "estimated": estimated_tokens,
                    "unit": "tokens"
                },
                {
                    "type": "daily_requests",
                    "current": usage_sum["daily_requests"],
                    "limit": limits.daily_request_limit,
                    "estimated": 1,
                    "unit": "requests"
                },
                {
                    "type": "monthly_requests",
                    "current": usage_sum["monthly_requests"],
                    "limit": limits.monthly_request_limit,
                    "estimated": 1,
                    "unit": "requests"
                }
            ]
            
            for check in checks:
                if check["limit"] > 0 and check["current"] + check["estimated"] > check["limit"]:
                    return {
                        "allowed": False,
                        "reason": f"{check['type']}_exceeded",
                        "limit": check["limit"],
                        "current": check["current"],
                        "remaining": max(0, check["limit"] - check["current"]),
                        "unit": check["unit"],
                        "message": f"You have exceeded your {check['type'].replace('_', ' ')} limit ({check['current']}/{check['limit']} {check['unit']})"
                    }
            
            return {"allowed": True}
            
        except Exception as e:
            log.error(f"Error checking user limits for user {user.id}: {e}")
            # 에러 발생 시 기본적으로 허용 (서비스 중단 방지)
            return {"allowed": True, "reason": "check_failed"}
    
    @staticmethod
    async def record_usage(user_id: str, url_idx: int, input_tokens: int, output_tokens: int, cost: float = 0.0):
        """API 사용량을 기록"""
        try:
            if not APIUsageTracker.is_external_api(url_idx):
                return
            
            total_tokens = input_tokens + output_tokens
            api_type = APIUsageTracker.get_api_type(url_idx)
            
            # 현재 사용량 레코드 조회 또는 생성
            usage = UserAPIUsageTable.get_user_usage(user_id, api_type, url_idx)
            if not usage:
                usage = UserAPIUsageTable.create_user_usage(user_id, api_type, url_idx)
            
            # 일일/월별 리셋 확인
            now = datetime.now()
            reset_daily = usage.last_daily_reset.date() < now.date()
            reset_monthly = (usage.last_monthly_reset.month != now.month or 
                           usage.last_monthly_reset.year != now.year)
            
            # 사용량 계산
            new_daily_tokens = total_tokens if reset_daily else usage.daily_tokens + total_tokens
            new_monthly_tokens = total_tokens if reset_monthly else usage.monthly_tokens + total_tokens
            new_daily_requests = 1 if reset_daily else usage.daily_requests + 1
            new_monthly_requests = 1 if reset_monthly else usage.monthly_requests + 1
            new_daily_cost = cost if reset_daily else usage.daily_cost + cost
            new_monthly_cost = cost if reset_monthly else usage.monthly_cost + cost
            
            # 레코드 업데이트
            UserAPIUsageTable.update_user_usage(
                usage.id,
                daily_tokens=new_daily_tokens,
                monthly_tokens=new_monthly_tokens,
                daily_requests=new_daily_requests,
                monthly_requests=new_monthly_requests,
                daily_cost=new_daily_cost,
                monthly_cost=new_monthly_cost,
                last_daily_reset=now if reset_daily else usage.last_daily_reset,
                last_monthly_reset=now if reset_monthly else usage.last_monthly_reset,
                updated_at=now
            )
            
            log.info(f"Recorded usage for user {user_id}: {total_tokens} tokens ({input_tokens} in + {output_tokens} out), cost: ${cost:.4f}")
            
        except Exception as e:
            log.error(f"Error recording usage for user {user_id}: {e}")
    
    @staticmethod
    def is_external_api(url_idx: int) -> bool:
        """외부 API인지 확인 (0번 인덱스는 보통 로컬 vLLM)"""
        # 0번은 로컬 vLLM으로 가정
        return url_idx > 1
    
    @staticmethod  
    def get_api_type(url_idx: int) -> str:
        """URL 인덱스로 API 타입 결정"""
        # 실제 환경에서는 request.app.state.config에서 URL을 확인해야 하지만
        # 여기서는 인덱스 기반으로 간단히 처리
        api_types = {
            0: "local",     # vLLM
            1: "local",     # rag API
            2: "groq",      # Groq API
            3: "openai"    # OpenAI API
        }
        return api_types.get(url_idx, f"external_{url_idx}")
    
    @staticmethod
    async def get_all_usage_stats() -> Dict:
        """전체 사용량 통계"""
        try:
            from open_webui.internal.db import get_db
            from sqlalchemy import func
            from open_webui.models.usage import UserAPIUsage
            
            with get_db() as db:
                # 오늘과 이번 달 기준점
                today = datetime.now().date()
                this_month_start = datetime.now().replace(day=1).date()
                
                # 전체 사용자 수
                total_users = db.query(func.count(func.distinct(UserAPIUsage.user_id))).scalar() or 0
                
                # 일일 통계 (오늘 사용량)
                daily_stats = db.query(
                    func.sum(UserAPIUsage.daily_tokens).label('tokens'),
                    func.sum(UserAPIUsage.daily_requests).label('requests'),
                    func.sum(UserAPIUsage.daily_cost).label('cost')
                ).filter(
                    func.date(UserAPIUsage.last_daily_reset) == today
                ).first()
                
                # 월별 통계 (이번 달 사용량)
                monthly_stats = db.query(
                    func.sum(UserAPIUsage.monthly_tokens).label('tokens'),
                    func.sum(UserAPIUsage.monthly_requests).label('requests'),
                    func.sum(UserAPIUsage.monthly_cost).label('cost')
                ).filter(
                    func.date(UserAPIUsage.last_monthly_reset) >= this_month_start
                ).first()
                
                # API별 사용량
                api_breakdown = db.query(
                    UserAPIUsage.api_type,
                    func.sum(UserAPIUsage.daily_tokens).label('daily_tokens'),
                    func.sum(UserAPIUsage.monthly_tokens).label('monthly_tokens')
                ).group_by(UserAPIUsage.api_type).all()
                
                # 상위 사용자 (월별 토큰 기준)
                top_users = db.query(
                    UserAPIUsage.user_id,
                    func.sum(UserAPIUsage.monthly_tokens).label('monthly_tokens'),
                    func.sum(UserAPIUsage.monthly_requests).label('monthly_requests')
                ).group_by(UserAPIUsage.user_id).order_by(
                    func.sum(UserAPIUsage.monthly_tokens).desc()
                ).limit(10).all()
                
                return {
                    "total_users": total_users,
                    "daily": {
                        "tokens": daily_stats.tokens or 0,
                        "requests": daily_stats.requests or 0,
                        "cost": float(daily_stats.cost or 0)
                    },
                    "monthly": {
                        "tokens": monthly_stats.tokens or 0,
                        "requests": monthly_stats.requests or 0,
                        "cost": float(monthly_stats.cost or 0)
                    },
                    "api_breakdown": {
                        api.api_type: {
                            "daily_tokens": api.daily_tokens or 0,
                            "monthly_tokens": api.monthly_tokens or 0
                        }
                        for api in api_breakdown
                    },
                    "top_users": [
                        {
                            "user_id": user.user_id,
                            "monthly_tokens": user.monthly_tokens or 0,
                            "monthly_requests": user.monthly_requests or 0
                        }
                        for user in top_users
                    ]
                }
                
        except Exception as e:
            log.error(f"Error getting usage stats: {e}")
            return {"error": str(e)}
    
    @staticmethod
    async def get_user_usage_details(user_id: str) -> Dict:
        """사용자별 사용량 상세"""
        try:
            usages = UserAPIUsageTable.get_all_user_usage(user_id)
            limits = UserAPILimitTable.get_user_limits(user_id)
            
            # 현재 시간 기준으로 리셋 여부 확인
            now = datetime.now()
            
            current_usage = []
            for usage in usages:
                daily_reset_needed = usage.last_daily_reset.date() < now.date()
                monthly_reset_needed = (usage.last_monthly_reset.month != now.month or 
                                      usage.last_monthly_reset.year != now.year)
                
                current_usage.append({
                    "api_type": usage.api_type,
                    "url_idx": usage.url_idx,
                    "daily_tokens": 0 if daily_reset_needed else usage.daily_tokens,
                    "monthly_tokens": 0 if monthly_reset_needed else usage.monthly_tokens,
                    "daily_requests": 0 if daily_reset_needed else usage.daily_requests,
                    "monthly_requests": 0 if monthly_reset_needed else usage.monthly_requests,
                    "daily_cost": 0.0 if daily_reset_needed else usage.daily_cost,
                    "monthly_cost": 0.0 if monthly_reset_needed else usage.monthly_cost,
                    "last_daily_reset": usage.last_daily_reset.isoformat(),
                    "last_monthly_reset": usage.last_monthly_reset.isoformat(),
                    "needs_daily_reset": daily_reset_needed,
                    "needs_monthly_reset": monthly_reset_needed
                })
            
            return {
                "user_id": user_id,
                "limits": {
                    "daily_token_limit": limits.daily_token_limit if limits else 50000,
                    "monthly_token_limit": limits.monthly_token_limit if limits else 500000,
                    "daily_request_limit": limits.daily_request_limit if limits else 200,
                    "monthly_request_limit": limits.monthly_request_limit if limits else 2000,
                    "daily_cost_limit": limits.daily_cost_limit if limits else 10.0,
                    "monthly_cost_limit": limits.monthly_cost_limit if limits else 100.0,
                    "enabled": limits.enabled if limits else False
                },
                "current_usage": current_usage,
                "total_apis": len(current_usage)
            }
            
        except Exception as e:
            log.error(f"Error getting user usage details for {user_id}: {e}")
            return {"user_id": user_id, "error": str(e)}
    
    @staticmethod
    async def set_user_limits(user_id: str, limits: Dict) -> Dict:
        """사용자별 제한 설정"""
        try:
            # 유효한 필드만 필터링
            valid_fields = {
                'daily_token_limit', 'monthly_token_limit',
                'daily_request_limit', 'monthly_request_limit',
                'daily_cost_limit', 'monthly_cost_limit', 'enabled'
            }
            filtered_limits = {k: v for k, v in limits.items() if k in valid_fields}
            
            updated_limits = UserAPILimitTable.update_user_limits(user_id, **filtered_limits)
            
            return {
                "success": True,
                "user_id": user_id,
                "limits": {
                    "daily_token_limit": updated_limits.daily_token_limit,
                    "monthly_token_limit": updated_limits.monthly_token_limit,
                    "daily_request_limit": updated_limits.daily_request_limit,
                    "monthly_request_limit": updated_limits.monthly_request_limit,
                    "daily_cost_limit": updated_limits.daily_cost_limit,
                    "monthly_cost_limit": updated_limits.monthly_cost_limit,
                    "enabled": updated_limits.enabled
                }
            }
        except Exception as e:
            log.error(f"Error setting user limits for {user_id}: {e}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    async def reset_user_usage(user_id: str, reset_type: str = "daily") -> Dict:
        """사용자 사용량 리셋 (관리자 기능)"""
        try:
            usages = UserAPIUsageTable.get_all_user_usage(user_id)
            reset_count = 0
            
            for usage in usages:
                if reset_type == "daily":
                    UserAPIUsageTable.update_user_usage(
                        usage.id,
                        daily_tokens=0,
                        daily_requests=0,
                        daily_cost=0.0,
                        last_daily_reset=datetime.now()
                    )
                elif reset_type == "monthly":
                    UserAPIUsageTable.update_user_usage(
                        usage.id,
                        monthly_tokens=0,
                        monthly_requests=0,
                        monthly_cost=0.0,
                        last_monthly_reset=datetime.now()
                    )
                elif reset_type == "all":
                    UserAPIUsageTable.update_user_usage(
                        usage.id,
                        daily_tokens=0,
                        monthly_tokens=0,
                        daily_requests=0,
                        monthly_requests=0,
                        daily_cost=0.0,
                        monthly_cost=0.0,
                        last_daily_reset=datetime.now(),
                        last_monthly_reset=datetime.now()
                    )
                reset_count += 1
            
            return {
                "success": True,
                "user_id": user_id,
                "reset_type": reset_type,
                "records_reset": reset_count
            }
            
        except Exception as e:
            log.error(f"Error resetting usage for {user_id}: {e}")
            return {"success": False, "error": str(e)}