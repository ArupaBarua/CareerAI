from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from sqlalchemy.ext.asyncio import AsyncSession
from src.components.auth.dependencies import get_current_user
from src.components.database.connection import get_db
from src.components.database.models import User
from src.components.resume.parser import extract_resume_text
from src.components.resume.schemas import ResumeResponse
from src.components.resume.service import (
    delete_resume,
    get_resume,
    get_user_resumes,
    save_resume,
)

router = APIRouter(
    prefix="/resumes",
    tags=["Resumes"]
)

@router.post(
    "",
    response_model=ResumeResponse,
    status_code=status.HTTP_201_CREATED
)
async def upload_resume(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF resumes are supported"
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded PDF is empty"
        )

    content = await extract_resume_text(file_bytes)

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not extract text from the resume"
        )

    return await save_resume(
        db=db,
        user_id=current_user.id,
        filename=file.filename or "resume.pdf",
        content=content
    )


@router.get(
    "",
    response_model=list[ResumeResponse]
)
async def list_resumes(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    return await get_user_resumes(
        db=db,
        user_id=current_user.id
    )


@router.delete(
    "/{resume_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def remove_resume(
    resume_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    resume = await get_resume(
        db=db,
        resume_id=resume_id,
        user_id=current_user.id,
    )

    if resume is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resume not found.",
        )

    await delete_resume(
        db=db,
        resume=resume,
    )