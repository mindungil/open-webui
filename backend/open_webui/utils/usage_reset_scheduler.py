import asyncio
import logging
from datetime import datetime, timezone, timedelta
from open_webui.models.usage import UserAPIUsageTable
from open_webui.internal.db import get_db
from sqlalchemy import func

log = logging.getLogger(__name__)

async def reset_daily_usage():
    """모든 사용자의 일일 사용량 초기화"""
    try:
        from open_webui.models.usage import UserAPIUsage

        with get_db() as db:
            now_utc = datetime.now(timezone.utc)

            # 모든 사용량 레코드 조회
            usages = db.query(UserAPIUsage).all()

            reset_count = 0
            for usage in usages:
                # last_daily_reset 확인
                last_daily = usage.last_daily_reset

                # 문자열인 경우 datetime으로 변환
                if isinstance(last_daily, str):
                    last_daily = datetime.fromisoformat(last_daily.replace('Z', '+00:00'))

                # naive datetime인 경우 UTC로 가정
                if last_daily.tzinfo is None:
                    last_daily = last_daily.replace(tzinfo=timezone.utc)

                # 날짜가 다르면 리셋 필요
                if last_daily.date() < now_utc.date():
                    usage.daily_tokens = 0
                    usage.daily_requests = 0
                    usage.daily_cost = 0.0
                    usage.last_daily_reset = now_utc
                    reset_count += 1

            if reset_count > 0:
                db.commit()
                log.info(f"Daily usage reset completed: {reset_count} records reset")
            else:
                log.info("Daily usage reset: No records needed reset")

    except Exception as e:
        log.error(f"Error in daily usage reset: {e}", exc_info=True)

async def reset_monthly_usage():
    """모든 사용자의 월간 사용량 초기화"""
    try:
        from open_webui.models.usage import UserAPIUsage

        with get_db() as db:
            now_utc = datetime.now(timezone.utc)

            # 모든 사용량 레코드 조회
            usages = db.query(UserAPIUsage).all()

            reset_count = 0
            for usage in usages:
                # last_monthly_reset 확인
                last_monthly = usage.last_monthly_reset

                # 문자열인 경우 datetime으로 변환
                if isinstance(last_monthly, str):
                    last_monthly = datetime.fromisoformat(last_monthly.replace('Z', '+00:00'))

                # naive datetime인 경우 UTC로 가정
                if last_monthly.tzinfo is None:
                    last_monthly = last_monthly.replace(tzinfo=timezone.utc)

                # 월이나 년도가 다르면 리셋 필요
                if last_monthly.month != now_utc.month or last_monthly.year != now_utc.year:
                    usage.monthly_tokens = 0
                    usage.monthly_requests = 0
                    usage.monthly_cost = 0.0
                    usage.last_monthly_reset = now_utc
                    reset_count += 1

            if reset_count > 0:
                db.commit()
                log.info(f"Monthly usage reset completed: {reset_count} records reset")
            else:
                log.info("Monthly usage reset: No records needed reset")

    except Exception as e:
        log.error(f"Error in monthly usage reset: {e}", exc_info=True)

async def reset_yearly_usage():
    """모든 사용자의 연간 사용량 초기화"""
    try:
        from open_webui.models.usage import UserAPIUsage

        with get_db() as db:
            now_utc = datetime.now(timezone.utc)

            # 모든 사용량 레코드 조회
            usages = db.query(UserAPIUsage).all()

            reset_count = 0
            for usage in usages:
                # last_yearly_reset 확인
                last_yearly = usage.last_yearly_reset if hasattr(usage, 'last_yearly_reset') and usage.last_yearly_reset else now_utc

                # 문자열인 경우 datetime으로 변환
                if isinstance(last_yearly, str):
                    last_yearly = datetime.fromisoformat(last_yearly.replace('Z', '+00:00'))

                # naive datetime인 경우 UTC로 가정
                if last_yearly.tzinfo is None:
                    last_yearly = last_yearly.replace(tzinfo=timezone.utc)

                # 년도가 다르면 리셋 필요
                if last_yearly.year < now_utc.year:
                    if hasattr(usage, 'yearly_tokens'):
                        usage.yearly_tokens = 0
                        usage.yearly_requests = 0
                        usage.yearly_cost = 0.0
                        usage.last_yearly_reset = now_utc
                        reset_count += 1

            if reset_count > 0:
                db.commit()
                log.info(f"Yearly usage reset completed: {reset_count} records reset")
            else:
                log.info("Yearly usage reset: No records needed reset")

    except Exception as e:
        log.error(f"Error in yearly usage reset: {e}", exc_info=True)

def get_seconds_until_next_midnight_utc():
    """다음 자정(UTC)까지 남은 초 계산"""
    now_utc = datetime.now(timezone.utc)
    tomorrow = now_utc.date() + timedelta(days=1)
    next_midnight = datetime.combine(tomorrow, datetime.min.time(), tzinfo=timezone.utc)
    return (next_midnight - now_utc).total_seconds()

def get_seconds_until_next_month():
    """다음 달 1일 00시(UTC)까지 남은 초 계산"""
    now_utc = datetime.now(timezone.utc)
    if now_utc.month == 12:
        next_month = datetime(now_utc.year + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    else:
        next_month = datetime(now_utc.year, now_utc.month + 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    return (next_month - now_utc).total_seconds()

def get_seconds_until_next_year():
    """다음 년도 1월 1일 00시(UTC)까지 남은 초 계산"""
    now_utc = datetime.now(timezone.utc)
    next_year = datetime(now_utc.year + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    return (next_year - now_utc).total_seconds()

async def periodic_usage_reset():
    """매일 자정(UTC)에 사용량 리셋 실행"""
    log.info("Starting periodic usage reset scheduler")

    while True:
        try:
            # 다음 자정까지 대기
            seconds_until_midnight = get_seconds_until_next_midnight_utc()
            log.info(f"Waiting {seconds_until_midnight/3600:.2f} hours until next daily reset (midnight UTC)")
            await asyncio.sleep(seconds_until_midnight + 10)  # 자정 후 10초에 실행

            # 일일 리셋 실행
            now_utc = datetime.now(timezone.utc)
            log.info(f"Running daily usage reset at {now_utc}")
            await reset_daily_usage()

            # 매달 1일이면 월간 리셋도 실행
            if now_utc.day == 1:
                log.info(f"Running monthly usage reset at {now_utc}")
                await reset_monthly_usage()

            # 매년 1월 1일이면 연간 리셋도 실행
            if now_utc.month == 1 and now_utc.day == 1:
                log.info(f"Running yearly usage reset at {now_utc}")
                await reset_yearly_usage()

        except Exception as e:
            log.error(f"Error in periodic usage reset: {e}", exc_info=True)
            # 에러 발생 시 1시간 후 재시도
            await asyncio.sleep(3600)
