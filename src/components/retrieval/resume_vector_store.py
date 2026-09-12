import asyncio
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.constants.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

embeddings = OpenAIEmbeddings(
    api_key=settings.OPENAI_API_KEY
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=200
)

def get_resume_index_path(
    user_id: int,
    resume_id: int
) -> Path:
    return Path(
        f"data/faiss/resumes/{user_id}/{resume_id}"
    )


async def create_resume_vector_store(
    resume_id: int,
    user_id: int,
    filename: str,
    content: str
) -> None:

    document = Document(
        page_content=content,
        metadata={
            "resume_id": resume_id,
            "user_id": user_id,
            "filename": filename
        }
    )

    chunks = text_splitter.split_documents([document])

    vector_store = await FAISS.afrom_documents(chunks, embeddings)

    index_path = get_resume_index_path(
        user_id=user_id,
        resume_id=resume_id
    )

    index_path.mkdir(
        parents=True,
        exist_ok=True
    )

    await asyncio.to_thread(
        vector_store.save_local,
        str(index_path)
    )