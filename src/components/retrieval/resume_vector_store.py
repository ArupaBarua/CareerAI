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
    chunk_size=1000,
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

    logger.info("Created FAISS vector store.")

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

    logger.info("Saved vector store.")


async def load_resume_vector_store(
    user_id: int,
    resume_id: int
) -> FAISS:
    index_path = get_resume_index_path(
        user_id=user_id,
        resume_id=resume_id
    )

    if not index_path.exists():
        logger.error(f"FAISS index not found for resume {resume_id}.")
        raise FileNotFoundError(
            f"FAISS index not found for resume {resume_id}."
        )

    vector_store = await asyncio.to_thread(
        FAISS.load_local,
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store


async def retrieve_resume_chunks(
    user_id: int,
    resume_id: int,
    query: str,
    k: int = 5,
) -> list[Document]:
    vector_store = await load_resume_vector_store(
        user_id=user_id,
        resume_id=resume_id,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
        },
    )

    documents = await retriever.ainvoke(query)

    return documents