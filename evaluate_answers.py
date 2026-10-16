
from pathlib import Path
import re
import unicodedata

import chromadb
import ollama


# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent

DB_DIR = str(PROJECT_DIR / "chroma_db")
COLLECTION_NAME = "business_reports"

CHAT_MODEL = "qwen2.5:3b"
EMBED_MODEL = "nomic-embed-text"

TEST_SOURCE = "sample_report.txt"
TOP_K = 4


# ==================================================
# CONNECT TO CHROMADB
# ==================================================

client = chromadb.PersistentClient(path=DB_DIR)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


# ==================================================
# TEST CASES
# ==================================================

TEST_CASES = [
    {
        "name": "Q3 revenue",
        "question": "What was the total revenue in Q3?",
        "required_terms": ["480000"],
        "must_abstain": False,
    },
    {
        "name": "Highest revenue region",
        "question": "Which region generated the highest revenue?",
        "required_terms": ["north", "190000"],
        "must_abstain": False,
    },
    {
        "name": "East region challenges",
        "question": "Why did the East region face challenges?",
        "required_terms": ["inventory shortages"],
        "must_abstain": False,
    },
    {
        "name": "Missing profit margin",
        "question": "What was the exact profit margin?",
        "required_terms": [],
        "must_abstain": True,
    },
]


ABSTENTION_PHRASES = [
    "couldn't find sufficient evidence",
    "cannot find sufficient evidence",
    "not provided",
    "does not provide",
    "not specified",
    "not available",
    "cannot determine",
    "insufficient information",
    "not stated",
]


# ==================================================
# TEXT NORMALIZATION
# ==================================================

def normalize_text(text):
    """
    Make checks insensitive to capitalization, punctuation,
    currency symbols, and thousands separators.
    """

    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = text.replace(",", "")

    return re.sub(r"\s+", " ", text).strip()


def normalize_number_and_words(text):
    """
    Keep letters and digits so '$480,000' matches '480000'.
    """

    text = normalize_text(text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


# ==================================================
# RETRIEVE EVIDENCE
# ==================================================

def retrieve_evidence(question, source_count):
    embedding_response = ollama.embed(
        model=EMBED_MODEL,
        input=question,
    )

    question_embedding = embedding_response["embeddings"][0]

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(TOP_K, source_count),
        where={"source": TEST_SOURCE},
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0] or []
    metadatas = results["metadatas"][0] or []

    context_parts = []

    for document, metadata in zip(documents, metadatas):
        source = metadata.get("source", TEST_SOURCE)
        location = metadata.get(
            "page",
            metadata.get("chunk", "unknown"),
        )

        context_parts.append(
            f"[Source: {source}, page/chunk: {location}]\n"
            f"{document}"
        )

    return documents, metadatas, "\n\n".join(context_parts)


# ==================================================
# GENERATE ANSWER
# ==================================================

def generate_answer(question, context):
    system_prompt = """
You are a careful business analyst.

Answer using only the provided evidence.

Rules:
- Never invent facts or numerical values.
- Compare relevant figures before selecting a highest or lowest value.
- If the evidence does not answer the question, say:
  "I couldn't find sufficient evidence in the provided documents."
- Cite the source filename and page or chunk identifier.
- Clearly distinguish facts from interpretations.
- Be concise and professional.
"""

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
                    f"Retrieved evidence:\n{context}\n\n"
                    f"Question: {question}"
                ),
            },
        ],
        options={"temperature": 0},
    )

    return response["message"]["content"].strip()


# ==================================================
# EVALUATE ONE ANSWER
# ==================================================

def evaluate_case(test, source_count):
    print("\n" + "=" * 65)
    print(f"TEST: {test['name']}")
    print(f"QUESTION: {test['question']}")

    documents, metadatas, context = retrieve_evidence(
        test["question"],
        source_count,
    )

    if not documents:
        answer = ""
        missing_terms = test["required_terms"]
        abstention_ok = test["must_abstain"]
        passed = False

        print("No evidence was retrieved.")
        print("RESULT: FAIL")

        return {
            "name": test["name"],
            "passed": passed,
            "missing_terms": missing_terms,
            "abstention_ok": abstention_ok,
        }

    answer = generate_answer(
        test["question"],
        context,
    )

    normalized_answer = normalize_number_and_words(answer)

    missing_terms = []

    for term in test["required_terms"]:
        normalized_term = normalize_number_and_words(term)

        if normalized_term not in normalized_answer:
            missing_terms.append(term)

    normalized_for_abstention = normalize_text(answer)

    abstained = any(
        phrase in normalized_for_abstention
        for phrase in ABSTENTION_PHRASES
    )

    if test["must_abstain"]:
        abstention_ok = abstained
    else:
        abstention_ok = not abstained

    passed = (
        not missing_terms
        and abstention_ok
    )

    print("\nGENERATED ANSWER:")
    print(answer)

    print("\nEVALUATION:")
    print(
        "Required facts:",
        "PASS" if not missing_terms else f"MISSING {missing_terms}",
    )

    if test["must_abstain"]:
        print(
            "Abstention:",
            "PASS" if abstention_ok else "FAIL",
        )

    else:
        print(
            "Unexpected abstention:",
            "NO" if abstention_ok else "YES",
        )

    print("\nRESULT:", "PASS" if passed else "FAIL")

    return {
        "name": test["name"],
        "passed": passed,
        "missing_terms": missing_terms,
        "abstention_ok": abstention_ok,
    }


# ==================================================
# RUN EVALUATION
# ==================================================

def main():
    stored = collection.get(
        where={"source": TEST_SOURCE},
        include=["metadatas"],
    )

    source_count = len(stored["ids"])

    print("=" * 65)
    print("NEXUS AI — ANSWER QUALITY EVALUATION")
    print("=" * 65)

    if source_count == 0:
        print(f"No indexed chunks found for {TEST_SOURCE}.")
        print("Upload and index the sample report first.")
        return

    print(f"Test source: {TEST_SOURCE}")
    print(f"Available chunks: {source_count}")

    results = []

    for test in TEST_CASES:
        try:
            result = evaluate_case(test, source_count)
            results.append(result)

        except Exception as error:
            print(f"\nERROR in {test['name']}: {error}")

            results.append({
                "name": test["name"],
                "passed": False,
                "missing_terms": [],
                "abstention_ok": False,
            })

    passed = sum(
        1 for result in results
        if result["passed"]
    )

    total = len(results)
    score = (passed / total * 100) if total else 0

    print("\n" + "=" * 65)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 65)

    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status}: {result['name']}")

    print(f"\nTests passed: {passed}/{total}")
    print(f"Answer check pass rate: {score:.1f}%")

    print(
        "\nNote: This is a small rule-based evaluation suite. "
        "Passing does not prove complete factual correctness "
        "or guarantee that every claim is grounded."
    )


if __name__ == "__main__":
    main()
