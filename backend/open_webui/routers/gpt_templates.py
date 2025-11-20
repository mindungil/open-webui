from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request

from open_webui.models.gpt_templates import (
    GPTTemplates,
    UserGPTTemplates,
    GPTTemplateModel,
    GPTTemplateForm,
    GPTTemplateUpdateForm,
    GPTTemplateUserResponse,
)
from open_webui.models.users import Users, UserResponse
from open_webui.constants import ERROR_MESSAGES
from open_webui.utils.auth import get_verified_user, get_admin_user
from open_webui.utils.access_control import has_access, has_permission

router = APIRouter()

############################
# GetGPTTemplates
############################


@router.get("/", response_model=list[GPTTemplateUserResponse])
async def get_gpt_templates(user=Depends(get_verified_user)):
    """공개된 모든 GPT 템플릿 조회"""
    templates = GPTTemplates.get_templates()

    # 사용자 정보 추가
    result = []
    for template in templates:
        template_user = Users.get_user_by_id(template.creator_id)
        result.append(
            GPTTemplateUserResponse(
                **template.model_dump(),
                user=UserResponse(**template_user.model_dump())
                if template_user
                else None,
            )
        )

    return result


############################
# GetFeaturedGPTTemplates
############################

## 필요 없음
# @router.get("/featured", response_model=list[GPTTemplateUserResponse])
# async def get_featured_gpt_templates(user=Depends(get_verified_user)):
#     """추천 GPT 템플릿만 조회"""
#     templates = GPTTemplates.get_featured_templates()

#     # 사용자 정보 추가
#     result = []
#     for template in templates:
#         template_user = Users.get_user_by_id(template.creator_id)
#         result.append(
#             GPTTemplateUserResponse(
#                 **template.model_dump(),
#                 user=UserResponse(**template_user.model_dump())
#                 if template_user
#                 else None,
#             )
#         )

#     return result


############################
# GetGPTTemplatesByCategory
############################

## 확장성 고려
# @router.get("/category/{category}", response_model=list[GPTTemplateUserResponse])
# async def get_gpt_templates_by_category(
#     category: str, user=Depends(get_verified_user)
# ):
#     """카테고리별 GPT 템플릿 조회"""
#     templates = GPTTemplates.get_templates_by_category(category)

#     # 사용자 정보 추가
#     result = []
#     for template in templates:
#         template_user = Users.get_user_by_id(template.creator_id)
#         result.append(
#             GPTTemplateUserResponse(
#                 **template.model_dump(),
#                 user=UserResponse(**template_user.model_dump())
#                 if template_user
#                 else None,
#             )
#         )

#     return result


############################
# CreateNewGPTTemplate
@router.post("/create", response_model=Optional[GPTTemplateModel])
async def create_gpt_template(
    request: Request, form_data: GPTTemplateForm, user=Depends(get_admin_user)
):

    """새 GPT 템플릿 생성"""

    # workspace.gpt_templates 권한 체크
    if user.role != "admin" and not has_permission(
        user.id, "workspace.gpt_templates", request.app.state.config.USER_PERMISSIONS
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )

    try:
        template = GPTTemplates.insert_new_template(user.id, form_data)
        if template:
            return template
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT("Failed to create GPT template"),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(str(e)),
        )


############################
# GetGPTTemplateById
############################


@router.get("/{id}", response_model=Optional[GPTTemplateModel])
async def get_gpt_template_by_id(id: str, user=Depends(get_verified_user)):
    """특정 GPT 템플릿 상세 조회"""
    template = GPTTemplates.get_template_by_id(id)

    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    # 공개 템플릿이 아니면 접근 권한 체크
    if not template.is_public:
        if user.role != "admin" and (
            user.id != template.creator_id
            and not has_access(user.id, type="read", access_control=template.access_control)
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
            )

    return template


############################
# UpdateGPTTemplateById
############################
@router.post("/{id}/update", response_model=Optional[GPTTemplateModel])
async def update_gpt_template_by_id(
    id: str, form_data: GPTTemplateUpdateForm, user=Depends(get_admin_user)):

    template = GPTTemplates.get_template_by_id(id)

    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    # 쓰기 권한 체크
    if user.role != "admin" and (
        user.id != template.creator_id
        and not has_access(user.id, type="write", access_control=template.access_control)
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    try:
        updated_template = GPTTemplates.update_template_by_id(id, form_data)
        if updated_template:
            return updated_template
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT("Failed to update GPT template"),
        )
    except Exception as e:
        raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(str(e)),
            )
        

############################
# DeleteGPTTemplateById
############################

@router.delete("/{id}/delete", response_model=bool)
async def delete_gpt_template_by_id(id: str, user=Depends(get_admin_user)):
#     """GPT 템플릿 삭제"""
    """GPT 템플릿 삭제"""
    template = GPTTemplates.get_template_by_id(id)

    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    result = GPTTemplates.delete_template_by_id(id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT("Failed to delete GPT template"),
        )

    return result


############################
# UseGPTTemplate
############################

## 현재로써는 필요 없는로직
## 템플릿 사용 시 단순히 사용 횟수만 증가시키는 로직
#
# @router.post("/{id}/use", response_model=bool)
# async def use_gpt_template(id: str, user=Depends(get_verified_user)):
#     """템플릿 사용 시 호출 (사용 횟수 증가)"""
#     template = GPTTemplates.get_template_by_id(id)

#     if not template:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail=ERROR_MESSAGES.NOT_FOUND,
#         )

#     # 공개 템플릿이 아니면 읽기 권한 체크
#     if not template.is_public:
#         if user.role != "admin" and (
#             user.id != template.creator_id
#             and not has_access(user.id, type="read", access_control=template.access_control)
#         ):
#             raise HTTPException(
#                 status_code=status.HTTP_403_FORBIDDEN,
#                 detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
#             )

#     result = GPTTemplates.increment_usage_count(id)
#     return result

############################
# AddGPTTemplateToUser
############################

@router.post("/{id}/add", response_model=bool)
async def add_gpt_template_to_user(id: str, user=Depends(get_verified_user)):
    """사용자에게 GPT 템플릿 활성화"""
    template = GPTTemplates.get_template_by_id(id)

    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    # 공개 템플릿이 아니면 읽기 권한 체크
    if not template.is_public:
        if user.role != "admin" and (
            user.id != template.creator_id
            and not has_access(user.id, type="read", access_control=template.access_control)
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
            )

    result = UserGPTTemplates.add_template_to_user(user.id, id)
    return result is not None


############################
# RemoveGPTTemplateFromUser
############################

@router.delete("/{id}/remove", response_model=bool)
async def remove_gpt_template_from_user(id: str, user=Depends(get_verified_user)):
    """사용자에게서 GPT 템플릿 비활성화"""
    result = UserGPTTemplates.remove_template_from_user(user.id, id)
    return result


############################
# GetUserGPTTemplates
############################

@router.get("/user/my", response_model=list[GPTTemplateUserResponse])
async def get_user_gpt_templates(user=Depends(get_verified_user)):
    """현재 사용자가 활성화한 GPT 템플릿 목록 조회"""
    templates = UserGPTTemplates.get_user_templates(user.id)

    # 사용자 정보 추가
    result = []
    for template in templates:
        template_user = Users.get_user_by_id(template.creator_id)
        result.append(
            GPTTemplateUserResponse(
                **template.model_dump(),
                user=UserResponse(**template_user.model_dump())
                if template_user
                else None,
            )
        )

    return result
