
from pathlib import Path

import chromadb
import ollama


# ==================================================
# 1. CONFIGURATION
# ==================================================

DATA_DIR = Path("data/sample_reports")
DB_DIR = "./chroma_db"

CHAT_MODEL = "qwen2.5:3b"
EMBED_MODEL = "nomic-embed-text"


# ==================================================
# 2. CONNECT TO CHROMADB
# ==================================================

client = chromadb.PersistentClient(path=DB_DIR)

collection = client.get_or_create_collection(
    name="business_reports",
    metadata={"hnsw:space": "cosine"},
)


# ==================================================
# 3. SPLIT TEXT INTO CHUNKS
# ==================================================

def chunk_text(text, chunk_size=120, overlap=25):
    """Split text into overlapping word-based chunks."""

    words = text.split()

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])

        if chunk:
            chunks.append(chunk)

    return chunks


# ==================================================
# 4. CREATE EMBEDDINGS
# ==================================================

def embed_texts(texts):
    """Convert text into numerical embeddings."""

    response = ollama.embed(
        model=EMBED_MODEL,
        input=texts,
    )

    return response["embeddings"]


# ==================================================
# 5. LOAD AND STORE DOCUMENTS
# ==================================================

def ingest_documents():
    """Read reports, create chunks, and store them."""

    files = list(DATA_DIR.glob("*.txt"))

    if not files:
        print(f"No .txt files found in {DATA_DIR.resolve()}")
        return

    total_chunks = 0

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        if not chunks:
            print(f"Skipping empty file: {file_path.name}")
            continue

        embeddings = embed_texts(chunks)

        ids = [
            f"{file_path.stem}_{i}"
            for i in range(len(chunks))
        ]

        metadatas = [
            {
                "source": file_path.name,
                "chunk": i,
            }
            for i in range(len(chunks))
        ]

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        total_chunks += len(chunks)

        print(
            f"Processed {file_path.name}: "
            f"{len(chunks)} chunks"
        )

    print(f"Total chunks processed: {total_chunks}")


# ==================================================
# 6. RETRIEVE EVIDENCE AND GENERATE AN ANSWER
# ==================================================

def answer_question(question, top_k=3):
    """Retrieve evidence and generate a grounded answer."""

    if collection.count() == 0:
        print("The database is empty.")
        print("Add a report and run ingestion first.")
        return

    # Embed the user's question
    question_embedding = embed_texts([question])[0]

    # Retrieve the most relevant document chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(top_k, collection.count()),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    # Extract retrieved passages and metadata
    documents = results["documents"][0] or []
    metadatas = results["metadatas"][0] or []
    distances = results["distances"][0] or []

    if not documents:
        print("No evidence was retrieved.")
        return

    # Build the context supplied to the language model
    context_parts = []

    for i, document in enumerate(documents):
        source = metadatas[i]["source"]
        chunk_number = metadatas[i]["chunk"]

        context_parts.append(
            f"[Source: {source}, chunk {chunk_number}]\n"
            f"{document}"
        )

    context = "\n\n".join(context_parts)

    # DEBUG: inspect what the retriever found
    print("\n" + "=" * 55)
    print("DEBUG — RETRIEVED EVIDENCE")
    print("=" * 55)

    for i, document in enumerate(documents):
        source = metadatas[i]["source"]
        chunk_number = metadatas[i]["chunk"]
        distance = distances[i]

        print(f"\nEvidence {i + 1}")
        print(f"Source: {source}")
        print(f"Chunk: {chunk_number}")
        print(f"Cosine distance: {distance:.4f}")
        print(document)

    print("\n" + "=" * 55)

    # Instructions for the language model
    system_prompt = """
You are a careful AI business analyst.

Answer using only the supplied evidence.

Rules:
1. Never invent facts or numerical values.
2. For comparison questions, compare all relevant values
   present in the evidence before choosing a winner.
3. Verify your conclusion against the supplied figures.
4. If the evidence does not answer the question, say:
   "I couldn't find sufficient evidence in the provided documents."
5. Cite the source filename and chunk identifier.
6. Clearly distinguish facts from interpretations.
7. If the evidence is incomplete, explain the limitation.
8. Keep answers concise and professional.
"""

    # Generate the answer locally with Ollama
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    "Retrieved evidence:\n"
                    f"{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer using only the evidence above."
                ),
            },
        ],
    )

    print("\nAI BUSINESS ANALYST")
    print("-" * 55)
    print(response["message"]["content"])
    print("-" * 55)


# ==================================================
# 7. RUN THE APPLICATION
# ==================================================

if __name__ == "__main__":

    # Check that the reports folder exists
    if not DATA_DIR.exists():
        print(f"Reports folder not found: {DATA_DIR.resolve()}")
        raise SystemExit(1)

    # Process documents before accepting questions
    ingest_documents()

    if collection.count() == 0:
        print("No documents are available to search.")
        raise SystemExit(1)

    # Interactive question-answer loop
    while True:
        question = input(
            "\nAsk a question (or type 'exit'): "
        ).strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            answer_question(question)

        except Exception as error:
            print(f"An error occurred: {error}")
            print(
                "Check that Ollama is running and both models "
                "are installed."
            )

