from langchain_openai import ChatOpenAI
from src.constants.settings import settings

llm = ChatOpenAI(
    model=settings.LLM_MODEL,
    api_key=settings.OPENAI_API_KEY,
    temperature=0
)

judge_llm = ChatOpenAI(
    model=settings.JUDGE_LLM_MODEL,
    api_key=settings.OPENAI_API_KEY,
    temperature=0
)