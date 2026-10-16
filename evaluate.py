
import chromadb
import ollama


# ==========================================
# CONFIGURATION
# ==========================================

DB_DIR = "./chroma_db"
COLLECTION_NAME = "business_reports"
EMBED_MODEL = "nomic-embed-text"

client = chromadb.PersistentClient(path=DB_DIR)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


# ==========================================
# TEST DATA
# ==========================================

TEST_CASES = [
    {
        "question": "What was the total revenue in Q3?",
        "expected_terms": ["480,000"],
    },
    {
        "question": "Which region generated the highest revenue?",
        "expected_terms": ["north", "190,000"],
    },
    {
        "question": "Why did the East region face challenges?",
        "expected_terms": ["east", "inventory shortages"],
    },
    {
        "question": "What was the exact profit margin?",
        "expected_terms": [
            "does not provide",
            "profit margins",
        ],
    },
]


# ==========================================
# EMBEDDINGS
# ==========================================

def embed_text(text):
    response = ollama.embed(
        model=EMBED_MODEL,
        input=text,
    )
    return response["embeddings"][0]


# ==========================================
# RETRIEVAL EVALUATION
# ==========================================

def evaluate_test(test, top_k=4):
    question = test["question"]
    expected_terms = test["expected_terms"]

    query_embedding = embed_text(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0] or []
    metadatas = results["metadatas"][0] or []

    retrieved_text = "\n".join(documents).lower()

    missing_terms = [
        term
        for term in expected_terms
        if term.lower() not in retrieved_text
    ]

    passed = len(missing_terms) == 0

    print("\n" + "=" * 65)
    print(f"QUESTION: {question}")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    print(f"Retrieved passages: {len(documents)}")

    if missing_terms:
        print("Expected evidence not found:", missing_terms)

    for index, (document, metadata) in enumerate(
        zip(documents, metadatas),
        start=1,
    ):
        print(
            f"\nPassage {index}: "
            f"{metadata.get('source', 'Unknown')}"
        )
        print(
            f"Page: {metadata.get('page', 'Unknown')}"
        )
        print(document[:350])

    return passed


# ==========================================
# RUN EVALUATION
# ==========================================

def main():
    if collection.count() == 0:
        print("No indexed documents found.")
        print("Upload and index a report before testing.")
        return

    print("NEXUS AI — RETRIEVAL EVALUATION")
    print(f"Stored chunks: {collection.count()}")

    passed = 0
    total = len(TEST_CASES)

    for test in TEST_CASES:
        try:
            if evaluate_test(test):
                passed += 1

        except Exception as error:
            print(f"\nTEST ERROR: {test['question']}")
            print(error)

    score = passed / total * 100

    print("\n" + "=" * 65)
    print("EVALUATION SUMMARY")
    print(f"Tests passed: {passed}/{total}")
    print(f"Evidence retrieval pass rate: {score:.1f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
