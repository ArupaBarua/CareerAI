from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Response,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from src.components.auth.schemas import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from src.components.auth.security import (
    create_access_token,
)
from src.components.auth.service import (
    authenticate_user,
    create_auth_session,
    create_user,
    get_user_by_username,
    revoke_session_by_refresh_token,
    rotate_refresh_token,
)
from src.components.database.connection import get_db
from src.components.database.models import User
from src.constants.settings import settings


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


def set_refresh_cookie(
    response: Response,
    refresh_token: str,
) -> None:
    max_age = (
        settings.REFRESH_COOKIE_MAX_AGE_DAYS
        * 24
        * 60
        * 60
    )

    response.set_cookie(
        key=settings.REFRESH_COOKIE_NAME,
        value=refresh_token,
        httponly=True,
        secure=settings.REFRESH_COOKIE_SECURE,
        samesite=settings.REFRESH_COOKIE_SAMESITE,
        max_age=max_age,
        path="/auth",
    )


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(
    request: RegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> User:
    existing_user = await get_user_by_username(
        db,
        request.username,
    )

    if existing_user is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists.",
        )

    return await create_user(
        db,
        request.username,
        request.password,
    )


@router.post(
    "/login",
    response_model=TokenResponse,
)
async def login(
    request: LoginRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    user = await authenticate_user(
        db,
        request.username,
        request.password,
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )

    auth_session, refresh_token = (
        await create_auth_session(
            db,
            user.id,
        )
    )

    access_token = create_access_token(
        user_id=user.id,
        session_id=auth_session.id,
    )

    set_refresh_cookie(
        response,
        refresh_token,
    )

    return TokenResponse(
        access_token=access_token
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
)
async def refresh(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    refresh_token = request.cookies.get(
        settings.REFRESH_COOKIE_NAME
    )

    if refresh_token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token missing.",
        )

    result = await rotate_refresh_token(
        db,
        refresh_token,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked refresh token.",
        )

    auth_session, new_refresh_token = result

    access_token = create_access_token(
        user_id=auth_session.user_id,
        session_id=auth_session.id,
    )

    set_refresh_cookie(
        response,
        new_refresh_token,
    )

    return TokenResponse(
        access_token=access_token
    )


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    refresh_token = request.cookies.get(
        settings.REFRESH_COOKIE_NAME
    )

    if refresh_token is not None:
        await revoke_session_by_refresh_token(
            db,
            refresh_token,
        )

    response.delete_cookie(
        key=settings.REFRESH_COOKIE_NAME,
        path="/auth",
    )

    return {
        "message": "Logged out successfully."
    }