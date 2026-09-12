from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.components.database.models import Resume


async def save_resume(
    db: AsyncSession,
    user_id: int,
    filename: str,
    content: str,
) -> Resume:
    resume = Resume(
        user_id=user_id,
        filename=filename,
        content=content,
    )

    db.add(resume)

    await db.commit()
    await db.refresh(resume)

    return resume


async def get_user_resumes(
    db: AsyncSession,
    user_id: int,
) -> list[Resume]:
    result = await db.execute(
        select(Resume)
        .where(
            Resume.user_id == user_id
        )
        .order_by(
            Resume.created_at.desc()
        )
    )

    return list(
        result.scalars().all()
    )


async def get_resume(
    db: AsyncSession,
    resume_id: int,
    user_id: int,
) -> Resume | None:
    result = await db.execute(
        select(Resume).where(
            Resume.id == resume_id,
            Resume.user_id == user_id,
        )
    )

    return result.scalar_one_or_none()


async def delete_resume(
    db: AsyncSession,
    resume: Resume,
) -> None:
    await db.delete(resume)
    await db.commit()