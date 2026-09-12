import os
import tempfile

from langchain_community.document_loaders import PyMuPDFLoader


async def extract_resume_text(
    file_bytes: bytes,
) -> str:
    temp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        loader = PyMuPDFLoader(temp_path)

        documents = await loader.aload()

        content = "\n".join(
            document.page_content
            for document in documents
        )

        return content.strip()

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)