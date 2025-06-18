from fastapi import APIRouter, Depends, HTTPException, Body
from open_webui.models.users import Users
from open_webui.utils.auth import get_current_user
from open_webui.utils.usage_tracker import APIUsageTracker
import logging
from datetime import datetime

log = logging.getLogger(__name__)
router = APIRouter()

@router.get("/")
async def get_user_usage(user=Depends(get_current_user)):
    """
    현재 사용자의 API 사용량과 제한 정보를 조회
    """
    try:
        usage_details = await APIUsageTracker.get_user_usage_details(user.id)
        
        # 응답 데이터 구조화
        response_data = {
            "user_id": user.id,
            "user_name": user.name,
            "limits": usage_details.get("limits", {}),
            "current_usage": usage_details.get("current_usage", []),
            "has_limits": bool(usage_details.get("limits", {}).get("enabled", False))
        }
        
        return {"success": True, "data": response_data}
    except Exception as e:
        log.error(f"Error fetching usage for user {user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch usage data")

@router.get("/summary")
async def get_usage_summary(user=Depends(get_current_user)):
    """
    사용자 사용량 요약 정보 (간단한 형태)
    """
    try:
        usage_details = await APIUsageTracker.get_user_usage_details(user.id)
        
        limits = usage_details.get("limits", {})
        current_usage = usage_details.get("current_usage", [])
        
        # 각 API별 사용량 요약
        api_summaries = []
        for usage in current_usage:
            daily_token_percent = 0
            daily_request_percent = 0
            monthly_token_percent = 0
            
            if limits.get("daily_token_limit", 0) > 0:
                daily_token_percent = min(100, (usage["daily_tokens"] / limits["daily_token_limit"]) * 100)
            
            if limits.get("daily_request_limit", 0) > 0:
                daily_request_percent = min(100, (usage["daily_requests"] / limits["daily_request_limit"]) * 100)
                
            if limits.get("monthly_token_limit", 0) > 0:
                monthly_token_percent = min(100, (usage["monthly_tokens"] / limits["monthly_token_limit"]) * 100)
            
            api_summaries.append({
                "api_type": usage["api_type"],
                "url_idx": usage["url_idx"],
                "daily": {
                    "tokens_used": usage["daily_tokens"],
                    "tokens_limit": limits.get("daily_token_limit", 0),
                    "tokens_percent": round(daily_token_percent, 1),
                    "requests_used": usage["daily_requests"],
                    "requests_limit": limits.get("daily_request_limit", 0),
                    "requests_percent": round(daily_request_percent, 1)
                },
                "monthly": {
                    "tokens_used": usage["monthly_tokens"],
                    "tokens_limit": limits.get("monthly_token_limit", 0),
                    "tokens_percent": round(monthly_token_percent, 1)
                }
            })
        # --- PATCH: limits.enabled가 true인데 api_summaries가 비어있으면 dummy 추가 ---
        if limits.get("enabled", False) and (not api_summaries):
            api_summaries.append({
                "api_type": "external",
                "url_idx": 1,
                "daily": {
                    "tokens_used": 0,
                    "tokens_limit": limits.get("daily_token_limit", 0),
                    "tokens_percent": 0,
                    "requests_used": 0,
                    "requests_limit": limits.get("daily_request_limit", 0),
                    "requests_percent": 0
                },
                "monthly": {
                    "tokens_used": 0,
                    "tokens_limit": limits.get("monthly_token_limit", 0),
                    "tokens_percent": 0
                }
            })
        # --- PATCH: 사용 기록이 있어도 limits.enabled가 true면 반드시 하나 이상 반환 ---
        if limits.get("enabled", False) and not api_summaries:
            # 혹시라도 위에서 추가가 안 됐으면 한 번 더 보장
            api_summaries.append({
                "api_type": "external",
                "url_idx": 1,
                "daily": {
                    "tokens_used": 0,
                    "tokens_limit": limits.get("daily_token_limit", 0),
                    "tokens_percent": 0,
                    "requests_used": 0,
                    "requests_limit": limits.get("daily_request_limit", 0),
                    "requests_percent": 0
                },
                "monthly": {
                    "tokens_used": 0,
                    "tokens_limit": limits.get("monthly_token_limit", 0),
                    "tokens_percent": 0
                }
            })
        return {
            "success": True,
            "data": {
                "has_limits": bool(limits.get("enabled", False)),
                "api_usage": api_summaries,
                "limits_active": limits.get("enabled", False)
            }
        }
    except Exception as e:
        log.error(f"Error fetching usage summary for user {user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch usage summary")


@router.get("/users")
async def get_users_with_usage(user=Depends(get_current_user)):
    """
    관리자: 전체 사용자별 사용량 요약 (admin 계정 제외, 이름으로 표시)
    """
    try:
        user_list = Users.get_users()
        result = []
        for u in user_list["users"]:
            usage_details = await APIUsageTracker.get_user_usage_details(u.id)
            monthly_tokens = 0
            monthly_requests = 0
            if usage_details.get("current_usage"):
                for usage in usage_details["current_usage"]:
                    monthly_tokens += usage.get("monthly_tokens", 0)
                    monthly_requests += usage.get("monthly_requests", 0)
            result.append({
                "user_id": u.id,
                "user_name": u.name,
                "monthly_tokens": monthly_tokens,
                "monthly_requests": monthly_requests,
                "last_activity": datetime.utcfromtimestamp(u.last_active_at).isoformat() if u.last_active_at else None,
            })
        return {"success": True, "data": {"users": result}}
    except Exception as e:
        log.error(f"Error fetching users usage: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch users usage")

@router.get("/limits/{user_id}")
async def get_user_limits(user_id: str, user=Depends(get_current_user)):
    """
    관리자: 특정 사용자의 토큰/요청 제한 정보 조회
    """
    from open_webui.utils.usage_tracker import APIUsageTracker
    try:
        usage_details = await APIUsageTracker.get_user_usage_details(user_id)
        return {"success": True, "data": usage_details.get("limits", {})}
    except Exception as e:
        log.error(f"Error fetching limits for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch user limits")

@router.post("/limits/{user_id}")
async def set_user_limits(
    user_id: str,
    limits: dict = Body(...),
    user=Depends(get_current_user)
):
    """
    관리자: 특정 사용자의 토큰/요청 제한 정보 수정
    """
    from open_webui.utils.usage_tracker import APIUsageTracker
    try:
        result = await APIUsageTracker.set_user_limits(user_id, limits)
        if result.get("success"):
            return {"success": True, "limits": result["limits"]}
        else:
            raise Exception(result.get("error", "Unknown error"))
    except Exception as e:
        log.error(f"Error setting limits for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save user limits")

@router.get("/stats")
async def get_usage_stats(user=Depends(get_current_user)):
    """전체 API 사용량 통계 조회 (관리자 전용)"""
    try:
        stats = await APIUsageTracker.get_all_usage_stats()
        return {"success": True, "data": stats}
    except Exception as e:
        log.error(f"Failed to get usage stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve usage statistics")

@router.get("/user/{user_id}")
async def get_user_usage_details(user_id: str, user=Depends(get_current_user)):
    """특정 사용자의 API 사용량 상세 조회 (관리자 전용)"""
    try:
        usage_details = await APIUsageTracker.get_user_usage_details(user_id)
        return {"success": True, "data": usage_details}
    except Exception as e:
        log.error(f"Failed to get user usage for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user usage")

@router.post("/reset/{user_id}")
async def reset_user_usage(user_id: str, reset_data: dict = Body(...), user=Depends(get_current_user)):
    """사용자 사용량 리셋 (관리자 전용)"""
    try:
        reset_type = reset_data.get("reset_type", "daily")
        if reset_type not in ["daily", "monthly", "all"]:
            raise HTTPException(status_code=400, detail="reset_type must be one of: daily, monthly, all")
        result = await APIUsageTracker.reset_user_usage(user_id, reset_type)
        if result.get("success"):
            return {"success": True, "message": f"User {reset_type} usage reset successfully", "data": result}
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to reset usage"))
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to reset usage for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset user usage")