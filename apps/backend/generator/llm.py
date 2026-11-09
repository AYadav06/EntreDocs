import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import settings
from generator import prompt
from langchain_openai import ChatOpenAI

generator = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=settings.OPENAI_API_KEY,
    temperature=0,
)


def answer(query: str, results: list, max_tokens: int = 512) -> str:
    """Generate a coherent answer from the retrieved search results.

    Args:
        query: The user's search question.
        results: List of Qdrant ScoredPoint objects returned by hybrid search.
        max_tokens: Maximum tokens for the generated response.

    Returns:
        The generated answer string.
    """
    if not results:
        return "No relevant documents were found for your query."

    prompt_text = prompt.build_prompt(query, results)
    response = generator.invoke(prompt_text)
    return response.content or "I was unable to generate an answer."


def main() -> None:
    """Test the generator with a sample query."""
    from retriever.Reranker import hybridSearch_with_rerank

    query = "How does sliding window attention work in Mistral 7B?"
    results = [p for p, _ in hybridSearch_with_rerank(query, k=3)]
    print(f"Query: {query}\n")
    print(f"Answer: {answer(query, results)}")


if __name__ == "__main__":
    main()
