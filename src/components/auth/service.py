from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.components.auth.security import (
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)
from src.components.database.models import (
    AuthSession,
    User,
)


async def get_user_by_username(
    db: AsyncSession,
    username: str,
) -> User | None:
    result = await db.execute(
        select(User).where(
            User.username == username
        )
    )

    return result.scalar_one_or_none()


async def create_user(
    db: AsyncSession,
    username: str,
    password: str,
) -> User:
    user = User(
        username=username,
        password_hash=hash_password(password),
    )

    db.add(user)

    await db.commit()
    await db.refresh(user)

    return user


async def authenticate_user(
    db: AsyncSession,
    username: str,
    password: str,
) -> User | None:
    user = await get_user_by_username(
        db,
        username,
    )

    if user is None:
        return None

    if not verify_password(
        password,
        user.password_hash,
    ):
        return None

    return user


async def create_auth_session(
    db: AsyncSession,
    user_id: int,
) -> tuple[AuthSession, str]:
    refresh_token = generate_refresh_token()

    auth_session = AuthSession(
        user_id=user_id,
        refresh_token_hash=hash_refresh_token(
            refresh_token
        ),
    )

    db.add(auth_session)

    await db.commit()
    await db.refresh(auth_session)

    return auth_session, refresh_token


async def get_active_session_by_id(
    db: AsyncSession,
    session_id: int,
) -> AuthSession | None:
    result = await db.execute(
        select(AuthSession).where(
            AuthSession.id == session_id,
            AuthSession.revoked_at.is_(None),
        )
    )

    return result.scalar_one_or_none()


async def rotate_refresh_token(
    db: AsyncSession,
    refresh_token: str,
) -> tuple[AuthSession, str] | None:
    token_hash = hash_refresh_token(
        refresh_token
    )

    result = await db.execute(
        select(AuthSession)
        .where(
            AuthSession.refresh_token_hash
            == token_hash,
            AuthSession.revoked_at.is_(None),
        )
        .with_for_update()
    )

    auth_session = result.scalar_one_or_none()

    if auth_session is None:
        return None

    new_refresh_token = generate_refresh_token()

    auth_session.refresh_token_hash = (
        hash_refresh_token(
            new_refresh_token
        )
    )

    auth_session.last_used_at = datetime.now(
        timezone.utc
    )

    await db.commit()

    return auth_session, new_refresh_token


async def revoke_session_by_refresh_token(
    db: AsyncSession,
    refresh_token: str,
) -> None:
    result = await db.execute(
        select(AuthSession).where(
            AuthSession.refresh_token_hash
            == hash_refresh_token(
                refresh_token
            ),
            AuthSession.revoked_at.is_(None),
        )
    )

    auth_session = result.scalar_one_or_none()

    if auth_session is None:
        return

    auth_session.revoked_at = datetime.now(
        timezone.utc
    )

    await db.commit()