from pydantic_settings import BaseSettings, SettingsConfigDict


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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()