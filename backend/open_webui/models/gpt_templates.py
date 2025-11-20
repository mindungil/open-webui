import time
import uuid
from typing import Optional
from pydantic import BaseModel, ConfigDict
from sqlalchemy import BigInteger, Boolean, Column, String, Text, JSON, Integer, Float, ForeignKey

from open_webui.internal.db import Base, get_db
from open_webui.models.users import Users, UserResponse
from open_webui.models.groups import Groups
from open_webui.utils.access_control import has_access

####################
# GPT Template DB Schema (전역 템플릿)
####################


class GPTTemplate(Base):
    __tablename__ = "gpt_templates"

    id = Column(Text, primary_key=True)
    creator_id = Column(Text)  # UUID 형태의 문자열

    # 기본 정보
    name = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    category = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)  # TEXT[] 대신 JSON 사용

    # 아이콘/이미지
    icon = Column(Text, nullable=True)
    banner_image = Column(Text, nullable=True)

    # GPT 설정
    system_prompt = Column(Text, nullable=True)
    conversation_starters = Column(JSON, nullable=True)

    # 기능 설정
    config = Column(JSON, nullable=True)
    capabilities = Column(JSON, nullable=True)
    knowledge_ids = Column(JSON, nullable=True)
    tool_ids = Column(JSON, nullable=True)

    # 공개 설정
    is_public = Column(Boolean, nullable=False, default=True)
    is_featured = Column(Boolean, default=False)

    # 통계
    usage_count = Column(BigInteger, nullable=False, default=0)
    rating_avg = Column(Float, nullable=True)  # NUMERIC(3,2)
    rating_count = Column(BigInteger, nullable=False, default=0)

    # 메타
    access_control = Column(JSON, nullable=True)
    meta = Column(JSON, nullable=True)

    # 타임스탬프 (TIMESTAMPTZ -> BigInteger로 epoch 저장)
    created_at = Column(BigInteger, nullable=False)
    # API 연동
    api_url = Column(Text, nullable=True)  # API 엔드포인트 (예: http://example.com:8080/v1)

    updated_at = Column(BigInteger, nullable=False)


class GPTTemplateModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    creator_id: str

    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[list] = None

    icon: Optional[str] = None
    banner_image: Optional[str] = None

    system_prompt: Optional[str] = None
    conversation_starters: Optional[list] = None

    config: Optional[dict] = None
    capabilities: Optional[dict] = None
    knowledge_ids: Optional[list] = None
    tool_ids: Optional[list] = None

    is_public: bool = True
    is_featured: bool = False

    usage_count: int = 0
    rating_avg: Optional[float] = None
    rating_count: int = 0

    access_control: Optional[dict] = None
    meta: Optional[dict] = None

    created_at: int
    updated_at: int

    # API 연동
    api_url: Optional[str] = None


####################
# User GPT Template DB Schema (사용자별 템플릿 사용 상태)
####################


class UserGPTTemplate(Base):
    __tablename__ = "user_gpt_templates"


    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(Text, nullable=False)  # UUID 형태의 문자열
    template_id = Column(Text, nullable=False)  # gpt_templates.id 참조

    # 사용중 여부
    is_using = Column(Boolean, nullable=False, default=True)

    # 타임스탬프
    created_at = Column(BigInteger, nullable=False)
    updated_at = Column(BigInteger, nullable=False)


class UserGPTTemplateModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: str
    template_id: str
    is_using: bool = True
    created_at: int
    updated_at: int


####################
# Forms
####################


class GPTTemplateForm(BaseModel):
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[list] = None

    icon: Optional[str] = None
    banner_image: Optional[str] = None

    system_prompt: Optional[str] = None
    conversation_starters: Optional[list] = None

    config: Optional[dict] = None
    capabilities: Optional[dict] = None
    knowledge_ids: Optional[list] = None
    tool_ids: Optional[list] = None

    is_public: bool = True
    is_featured: bool = False

    access_control: Optional[dict] = None
    meta: Optional[dict] = None

    # API 연동
    api_url: Optional[str] = None


class GPTTemplateUpdateForm(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[list] = None

    icon: Optional[str] = None
    banner_image: Optional[str] = None

    system_prompt: Optional[str] = None
    conversation_starters: Optional[list] = None

    config: Optional[dict] = None
    capabilities: Optional[dict] = None
    knowledge_ids: Optional[list] = None
    tool_ids: Optional[list] = None

    is_public: Optional[bool] = None
    is_featured: Optional[bool] = None

    access_control: Optional[dict] = None
    meta: Optional[dict] = None

    # API 연동
    api_url: Optional[str] = None


class GPTTemplateUserResponse(GPTTemplateModel):
    user: Optional[UserResponse] = None


class UserGPTTemplateForm(BaseModel):
    template_id: str


####################
# GPTTemplateTable
####################


class GPTTemplateTable:
    def insert_new_template(
        self, creator_id: str, form_data: GPTTemplateForm
    ) -> Optional[GPTTemplateModel]:
        with get_db() as db:
            template = GPTTemplateModel(
                **{
                    "id": str(uuid.uuid4()),
                    "creator_id": creator_id,
                    **form_data.model_dump(),
                    "usage_count": 0,
                    "rating_avg": None,
                    "rating_count": 0,
                    "created_at": int(time.time()),
                    "updated_at": int(time.time()),
                }
            )

            try:
                result = GPTTemplate(**template.model_dump())
                db.add(result)
                db.commit()
                db.refresh(result)
                return GPTTemplateModel.model_validate(result) if result else None
            except Exception as e:
                print(f"Error creating GPT template: {e}")
                return None

    def get_templates(self) -> list[GPTTemplateModel]:
        """공개된 모든 템플릿 조회"""
        with get_db() as db:
            templates = (
                db.query(GPTTemplate)
                .filter(GPTTemplate.is_public == True)
                .order_by(GPTTemplate.updated_at.desc())
                .all()
            )
            return [GPTTemplateModel.model_validate(template) for template in templates]

    def get_featured_templates(self) -> list[GPTTemplateModel]:
        """추천 템플릿만 조회"""
        with get_db() as db:
            templates = (
                db.query(GPTTemplate)
                .filter(GPTTemplate.is_public == True, GPTTemplate.is_featured == True)
                .order_by(GPTTemplate.updated_at.desc())
                .all()
            )
            return [GPTTemplateModel.model_validate(template) for template in templates]

    def get_templates_by_category(self, category: str) -> list[GPTTemplateModel]:
        """카테고리별 템플릿 조회"""
        with get_db() as db:
            templates = (
                db.query(GPTTemplate)
                .filter(
                    GPTTemplate.is_public == True, GPTTemplate.category == category
                )
                .order_by(GPTTemplate.updated_at.desc())
                .all()
            )
            return [GPTTemplateModel.model_validate(template) for template in templates]

    def get_templates_by_creator_id(
        self, creator_id: str, permission: str = "read"
    ) -> list[GPTTemplateModel]:
        """생성자 ID로 템플릿 조회 (권한 체크 포함)"""
        with get_db() as db:
            user_group_ids = {
                group.id for group in Groups.get_groups_by_member_id(creator_id)
            }

            templates = db.query(GPTTemplate).order_by(GPTTemplate.updated_at.desc()).all()

            result = []
            for template in templates:
                if template.creator_id == creator_id or has_access(
                    creator_id,
                    permission,
                    template.access_control,
                    user_group_ids,
                ):
                    result.append(GPTTemplateModel.model_validate(template))

            return result

    def get_template_by_id(self, id: str) -> Optional[GPTTemplateModel]:
        """ID로 템플릿 조회"""
        with get_db() as db:
            template = db.query(GPTTemplate).filter(GPTTemplate.id == id).first()
            return GPTTemplateModel.model_validate(template) if template else None

    def update_template_by_id(
        self, id: str, form_data: GPTTemplateUpdateForm
    ) -> Optional[GPTTemplateModel]:
        """템플릿 수정"""
        with get_db() as db:
            template = db.query(GPTTemplate).filter(GPTTemplate.id == id).first()
            if not template:
                return None

            update_data = form_data.model_dump(exclude_unset=True)
            for key, value in update_data.items():
                setattr(template, key, value)

            template.updated_at = int(time.time())
            db.commit()
            db.refresh(template)

            return GPTTemplateModel.model_validate(template)

    def delete_template_by_id(self, id: str) -> bool:
        """템플릿 삭제"""
        try:
            with get_db() as db:
                db.query(GPTTemplate).filter(GPTTemplate.id == id).delete()
                db.commit()
                return True
        except Exception as e:
            print(f"Error deleting GPT template: {e}")
            return False

    def increment_usage_count(self, id: str) -> bool:
        """사용 횟수 증가"""
        try:
            with get_db() as db:
                template = db.query(GPTTemplate).filter(GPTTemplate.id == id).first()
                if template:
                    template.usage_count += 1
                    db.commit()
                    return True
                return False
        except Exception as e:
            print(f"Error incrementing usage count: {e}")
            return False


####################
# UserGPTTemplateTable
####################


class UserGPTTemplateTable:
    def add_template_to_user(
        self, user_id: str, template_id: str
    ) -> Optional[UserGPTTemplateModel]:
        """사용자에게 템플릿 추가 (내 템플릿으로 등록)"""
        with get_db() as db:
            # 이미 존재하는지 확인
            existing = (
                db.query(UserGPTTemplate)
                .filter(
                    UserGPTTemplate.user_id == user_id,
                    UserGPTTemplate.template_id == template_id,
                )
                .first()
            )

            if existing:
                # 이미 있으면 is_using을 true로 업데이트
                existing.is_using = True
                existing.updated_at = int(time.time())
                db.commit()
                db.refresh(existing)
                return UserGPTTemplateModel.model_validate(existing)

            # 없으면 새로 생성
            user_template = UserGPTTemplateModel(
                **{
                    "id": 0,  # autoincrement
                    "user_id": user_id,
                    "template_id": template_id,
                    "is_using": True,
                    "created_at": int(time.time()),
                    "updated_at": int(time.time()),
                }
            )

            try:
                result = UserGPTTemplate(**user_template.model_dump(exclude={"id"}))
                db.add(result)
                db.commit()
                db.refresh(result)
                return UserGPTTemplateModel.model_validate(result) if result else None
            except Exception as e:
                print(f"Error adding template to user: {e}")
                return None

    def get_user_templates(self, user_id: str) -> list[GPTTemplateModel]:
        """사용자가 추가한 템플릿 목록 조회 (is_using=True인 것만)"""
        with get_db() as db:
            user_templates = (
                db.query(UserGPTTemplate)
                .filter(
                    UserGPTTemplate.user_id == user_id,
                    UserGPTTemplate.is_using == True,
                )
                .all()
            )

            template_ids = [ut.template_id for ut in user_templates]

            if not template_ids:
                return []

            templates = (
                db.query(GPTTemplate)
                .filter(GPTTemplate.id.in_(template_ids))
                .all()
            )

            return [GPTTemplateModel.model_validate(t) for t in templates]

    def remove_template_from_user(self, user_id: str, template_id: str) -> bool:
        """사용자의 템플릿 제거 (is_using을 False로)"""
        try:
            with get_db() as db:
                user_template = (
                    db.query(UserGPTTemplate)
                    .filter(
                        UserGPTTemplate.user_id == user_id,
                        UserGPTTemplate.template_id == template_id,
                    )
                    .first()
                )

                if user_template:
                    user_template.is_using = False
                    user_template.updated_at = int(time.time())
                    db.commit()
                    return True
                return False
        except Exception as e:
            print(f"Error removing template from user: {e}")
            return False

    def is_template_added(self, user_id: str, template_id: str) -> bool:
        """사용자가 해당 템플릿을 추가했는지 확인"""
        with get_db() as db:
            user_template = (
                db.query(UserGPTTemplate)
                .filter(
                    UserGPTTemplate.user_id == user_id,
                    UserGPTTemplate.template_id == template_id,
                    UserGPTTemplate.is_using == True,
                )
                .first()
            )
            return user_template is not None


GPTTemplates = GPTTemplateTable()
UserGPTTemplates = UserGPTTemplateTable()
