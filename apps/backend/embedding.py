from langchain_openai import OpenAIEmbeddings

from config.config import settings

OPENAI_API_KEY = settings.OPENAI_API_KEY

# Single source of truth: the model AND its output dimension live together.
# text-embedding-3-large -> 3072, text-embedding-3-small -> 1536, ada-002 -> 1536.
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=OPENAI_API_KEY,
    # Force the model's output to match EMBEDDING_DIMENSIONS so they can't drift.
    dimensions=EMBEDDING_DIMENSIONS,
)
