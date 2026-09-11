from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "CareerAI"
    APP_VERSION: str = "1.0.0"
    LOG_LEVEL: str = "INFO"

    DATABASE_URL: str

    OPENAI_API_KEY: str

    LANGSMITH_API_KEY: str
    LANGSMITH_TRACING: bool
    LANGSMITH_PROJECT: str

    LLM_MODEL: str = "gpt-4o-mini"
    JUDGE_LLM_MODEL: str = "gpt-4o"

    EXA_MCP_URL: str
    WORKOPIA_MCP_URL: str

    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    REFRESH_COOKIE_NAME: str = "careerai_refresh_token"
    REFRESH_COOKIE_MAX_AGE_DAYS: int = 365
    REFRESH_COOKIE_SECURE: bool = False
    REFRESH_COOKIE_SAMESITE: Literal[
        "lax",
        "strict",
        "none",
    ] = "lax"

    FRONTEND_ORIGIN: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()