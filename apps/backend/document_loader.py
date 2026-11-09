from pathlib import Path

import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vectorStore.qdrant import init_vector_store


def load_pdf_pages(file_path: str) -> list[Document]:

    reader = pypdf.PdfReader(file_path)
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": file_path, "page": i},
        )
        for i, page in enumerate(reader.pages)
    ]


def load_documents(folder_path: str) -> list[Document]:
    """Load every PDF in a folder into one flat list of page-level Documents."""
    folder = Path(folder_path)
    docs: list[Document] = []
    for pdf_path in sorted(folder.glob("*.pdf")):
        print(f"Loading {pdf_path.name} ...")
        docs.extend(load_pdf_pages(str(pdf_path)))
    return docs


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    documents_dir = base_dir / "documents"

    docs = load_documents(str(documents_dir))
    print(f"Loaded {len(docs)} pages from {documents_dir}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"{len(all_splits)} chunks")

    # QdrantVectorStore embeddings.
    vector_store = init_vector_store()
    vector_store.add_documents(documents=all_splits)
    print(f"indexed {len(all_splits)} chunks")


if __name__ == "__main__":
    main()
