from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config=SettingsConfigDict(env_file=".env")

    OPENAI_API_KEY:str
    GROQ_API_KEY:str
    QDRANT_API_KEY:str
    QDRANT_ENDPOINT:str

settings=Settings()
