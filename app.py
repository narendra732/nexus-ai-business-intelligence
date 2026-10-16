
from pathlib import Path
import hashlib
import html

import chromadb
import ollama
import streamlit as st
from pypdf import PdfReader

from business_rules import deterministic_business_answer


# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Nexus AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_DIR = Path(__file__).resolve().parent
DB_DIR = str(PROJECT_DIR / "chroma_db")
UPLOAD_DIR = PROJECT_DIR / "data" / "uploaded_reports"

CHAT_MODEL = "qwen2.5:3b"
EMBED_MODEL = "nomic-embed-text"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

client = chromadb.PersistentClient(path=DB_DIR)

collection = client.get_or_create_collection(
    name="business_reports",
    metadata={"hnsw:space": "cosine"},
)


# =========================================================
# 2. DESIGN
# =========================================================

st.markdown(
    """
    <style>
    @import url(
      'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap'
    );

    :root {
        --ink: #262832;
        --muted: #7b8190;
        --line: #e8e9ee;
        --accent: #5b5ce2;
        --sidebar-bg: #f8f8fa;
    }

    .stApp {
        background: #ffffff;
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        letter-spacing: -0.6px;
    }

    .block-container {
        max-width: 1000px;
        padding-top: 1.25rem;
        padding-bottom: 5rem;
    }

    /* Keep the native Streamlit header and sidebar toggle. */
    header[data-testid="stHeader"] {
        display: block !important;
        visibility: visible !important;
        background: #ffffff !important;
    }

    [data-testid="stHeader"] button {
        visibility: visible !important;
        opacity: 1 !important;
    }

    [data-testid="stToolbar"] {
        visibility: visible !important;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--sidebar-bg);
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 1rem 0.8rem;
        box-sizing: border-box;
    }

    [data-testid="stSidebar"] .stButton > button {
        border-radius: 10px;
        min-height: 40px;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: #ffffff;
        border: 1px dashed #d0d3de;
        border-radius: 12px;
    }

    .brand {
        display: grid !important;
        grid-template-columns: 42px minmax(0, 1fr);
        align-items: center;
        column-gap: 10px;
        width: 100%;
        box-sizing: border-box;
        padding: 8px 0 20px;
    }

    .brand-icon {
        width: 42px;
        height: 42px;
        box-sizing: border-box;
        border-radius: 12px;
        background: linear-gradient(135deg, #7778ff, #5253d5);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #ffffff;
        font-size: 24px;
        font-weight: 800;
        flex-shrink: 0;
    }

    .brand-copy {
        min-width: 0;
    }

    .brand-name {
        font-family: 'Manrope', sans-serif;
        font-size: 18px;
        line-height: 1.35;
        font-weight: 800;
        letter-spacing: -0.6px;
        color: #252832;
        white-space: nowrap;
    }

    .brand-subtitle {
        margin-top: 3px;
        font-size: 8px;
        line-height: 1.5;
        color: #858b99;
        letter-spacing: 1px;
        font-weight: 700;
        white-space: nowrap;
    }

    .side-label {
        font-size: 10px;
        letter-spacing: 1.4px;
        font-weight: 700;
        color: #8b909c;
        margin: 18px 0 10px;
    }

    .sidebar-note {
        border: 1px solid #e7e7f5;
        background: #f1f1ff;
        border-radius: 11px;
        padding: 12px;
        color: #64668b;
        font-size: 11px;
        line-height: 1.7;
        margin-top: 15px;
    }

    /* Main header */
    .topbar-title {
        font-family: 'Manrope', sans-serif;
        font-size: 17px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #252832;
    }

    .topbar-subtitle {
        font-size: 11px;
        margin-top: 3px;
        color: var(--muted);
    }

    .local-pill {
        display: inline-block;
        padding: 7px 10px;
        color: #137d51;
        background: #e9f8ef;
        border-radius: 20px;
        font-size: 10px;
        font-weight: 700;
        white-space: nowrap;
    }

    /* Welcome screen */
    .welcome {
        text-align: center;
        padding: 48px 0 26px;
    }

    .welcome-icon {
        width: 58px;
        height: 58px;
        margin: 0 auto 23px;
        border-radius: 19px;
        background: #efefff;
        color: #5b5ce2;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 31px;
        font-weight: 800;
    }

    .welcome-title {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(27px, 4vw, 39px);
        line-height: 1.25;
        font-weight: 800;
        letter-spacing: -1.4px;
        color: #272a34;
        margin-bottom: 14px;
    }

    .welcome-text {
        font-size: 14px;
        line-height: 1.9;
        color: var(--muted);
        max-width: 535px;
        margin: 0 auto;
    }

    .suggestion-card {
        border: 1px solid var(--line);
        background: #ffffff;
        border-radius: 13px;
        padding: 15px;
        min-height: 90px;
        margin-bottom: 7px;
    }

    .suggestion-title {
        color: #303440;
        font-weight: 700;
        font-size: 13px;
        margin-bottom: 6px;
    }

    .suggestion-text {
        font-size: 12px;
        color: #858b98;
        line-height: 1.7;
    }

    /* Chat */
    [data-testid="stChatMessage"] {
        background: #ffffff;
        border: 1px solid #ebecf0;
        border-radius: 14px;
        padding: 15px 17px;
        margin-bottom: 13px;
    }

    [data-testid="stChatMessage"] p {
        line-height: 1.8;
    }

    [data-testid="stChatInput"] {
        background: #ffffff;
        border: 1px solid #dfe1e8;
        border-radius: 16px;
        box-shadow: 0 6px 25px rgba(20, 25, 45, 0.045);
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: #aaaaf3;
        box-shadow: 0 0 0 3px rgba(91, 92, 226, 0.08);
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.15s ease;
    }

    .stButton > button[kind="primary"] {
        background: var(--accent);
        color: #ffffff;
        border: 1px solid var(--accent);
    }

    .stButton > button[kind="primary"]:hover {
        background: #4849c7;
        border-color: #4849c7;
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--line);
        border-radius: 11px;
        background: #ffffff;
    }

    .source-card {
        border: 1px solid #e8e9ee;
        border-radius: 10px;
        padding: 12px;
        margin: 8px 0 13px;
        background: #fafbfc;
        color: #4e5564;
        font-size: 13px;
        line-height: 1.8;
        overflow-wrap: anywhere;
    }

    .footer-note {
        text-align: center;
        color: #9298a6;
        font-size: 11px;
        line-height: 1.7;
        padding-top: 22px;
    }

    @media (max-width: 700px) {
        .block-container {
            padding: 1rem 1rem 5rem;
        }

        .welcome {
            padding-top: 24px;
        }

        .welcome-title {
            letter-spacing: -0.8px;
        }

        .brand-subtitle {
            letter-spacing: 0.6px;
            font-size: 7px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. DOCUMENT PROCESSING
# =========================================================

def chunk_text(text, chunk_size=120, overlap=25):
    words = text.split()
    step = chunk_size - overlap

    if step <= 0:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size."
        )

    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), step)
        if words[i:i + chunk_size]
    ]


def read_document(path):
    pages = []

    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                pages.append((page_number, text))

    elif path.suffix.lower() == ".txt":
        text = path.read_text(encoding="utf-8")

        if text.strip():
            pages.append((1, text))

    return pages


def embed_texts(texts):
    response = ollama.embed(
        model=EMBED_MODEL,
        input=texts,
    )
    return response["embeddings"]


def index_document(path):
    pages = read_document(path)

    chunks = []
    metadatas = []

    for page_number, text in pages:
        for chunk in chunk_text(text):
            chunks.append(chunk)
            metadatas.append({
                "source": path.name,
                "page": page_number,
            })

    if not chunks:
        return 0

    file_key = hashlib.sha256(
        path.name.encode("utf-8")
    ).hexdigest()[:12]

    ids = [
        f"{file_key}_{i}"
        for i in range(len(chunks))
    ]

    embeddings = embed_texts(chunks)

    existing = collection.get(
        where={"source": path.name},
        include=["metadatas"],
    )

    old_ids = existing["ids"]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    obsolete_ids = [
        old_id for old_id in old_ids
        if old_id not in ids
    ]

    if obsolete_ids:
        collection.delete(ids=obsolete_ids)

    return len(chunks)


def get_document_library():
    result = collection.get(include=["metadatas"])
    library = {}

    for metadata in result["metadatas"] or []:
        source = metadata.get("source", "Unknown")

        if source not in library:
            library[source] = {
                "chunks": 0,
                "pages": set(),
            }

        library[source]["chunks"] += 1

        page = metadata.get("page")

        if page is not None:
            library[source]["pages"].add(page)

    return library


def remove_document(source):
    result = collection.get(
        where={"source": source},
        include=["metadatas"],
    )

    ids = result["ids"]

    if not ids:
        return 0

    collection.delete(ids=ids)

    uploaded_copy = UPLOAD_DIR / Path(source).name

    if uploaded_copy.exists():
        uploaded_copy.unlink()

    return len(ids)


# =========================================================
# 4. RAG QUESTION ANSWERING
# =========================================================

def answer_question(question, top_k=4):
    """Retrieve relevant evidence, then answer the question."""

    if collection.count() == 0:
        return (
            "Please upload and index a report before asking questions.",
            [],
        )

    question_embedding = embed_texts([question])[0]

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(top_k, collection.count()),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = results["documents"][0] or []
    metadatas = results["metadatas"][0] or []
    distances = results["distances"][0] or []

    if not documents:
        return (
            "I couldn't find sufficient evidence in the documents.",
            [],
        )

    citations = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        citations.append({
            "source": metadata.get("source", "Unknown"),
            "page": metadata.get(
                "page",
                metadata.get("chunk", "Unknown"),
            ),
            "text": document,
            "distance": distance,
        })

    # Use the shared Python business rule for recognized
    # numerical ranking questions. Inspect all indexed
    # passages instead of relying only on top-k retrieval.
    all_records = collection.get(
        include=["documents", "metadatas"],
    )

    all_documents = all_records["documents"] or []
    all_metadatas = all_records["metadatas"] or []

    rule_answer = deterministic_business_answer(
        question,
        all_documents,
        all_metadatas,
    )

    if rule_answer is not None:
        return rule_answer, citations

    # Normal RAG answer generation.
    context_parts = []

    for document, metadata in zip(documents, metadatas):
        source = metadata.get("source", "Unknown")
        location = metadata.get(
            "page",
            metadata.get("chunk", "Unknown"),
        )

        context_parts.append(
            f"[Source: {source}, page/chunk: {location}]\n"
            f"{document}"
        )

    context = "\n\n".join(context_parts)

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": """
You are Nexus AI, a careful business analyst.

Answer using only the supplied evidence.

- Never invent facts, numbers, or explanations.
- Compare relevant figures before drawing conclusions.
- If evidence is insufficient, say so clearly.
- Cite source filenames and page/chunk identifiers.
- Distinguish facts from interpretations.
- Be concise and professional.
""",
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

    return response["message"]["content"].strip(), citations


# =========================================================
# 5. SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# =========================================================
# 6. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
          <div class="brand-icon">✦</div>
          <div class="brand-copy">
            <div class="brand-name">NEXUS AI</div>
            <div class="brand-subtitle">INTELLIGENCE WORKSPACE</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("＋  New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.markdown(
        '<div class="side-label">YOUR DOCUMENTS</div>',
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT reports",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button(
        "＋  Index documents",
        type="primary",
        use_container_width=True,
    ):
        if not uploaded_files:
            st.warning("Please upload at least one document.")

        else:
            progress = st.progress(0)
            indexed_count = 0

            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    safe_name = Path(uploaded_file.name).name
                    destination = UPLOAD_DIR / safe_name

                    destination.write_bytes(
                        uploaded_file.getvalue()
                    )

                    chunk_count = index_document(destination)

                    if chunk_count > 0:
                        indexed_count += 1
                    else:
                        st.warning(
                            f"No extractable text found in {safe_name}."
                        )

                except Exception as error:
                    st.error(
                        f"Could not index {uploaded_file.name}: {error}"
                    )

                progress.progress(
                    (i + 1) / len(uploaded_files)
                )

            if indexed_count:
                st.success(
                    f"Indexed {indexed_count} document(s)."
                )

            st.rerun()

    library = get_document_library()

    st.caption(
        f"{len(library)} documents · {collection.count()} chunks"
    )

    if library:
        for source, details in sorted(library.items()):
            with st.expander(f"📄 {source}"):
                st.caption(
                    f"{details['chunks']} text chunks"
                )

                if details["pages"]:
                    st.caption(
                        f"{len(details['pages'])} pages represented"
                    )

        with st.expander("Manage documents"):
            selected_source = st.selectbox(
                "Select document to remove",
                options=sorted(library.keys()),
                key="remove_document_selection",
            )

            confirm_remove = st.checkbox(
                "Confirm document removal",
                key="confirm_remove_document",
            )

            if st.button(
                "Remove selected document",
                disabled=not confirm_remove,
                use_container_width=True,
            ):
                deleted_count = remove_document(selected_source)

                if deleted_count:
                    st.success(
                        f"Removed {deleted_count} stored chunks."
                    )
                    st.rerun()
                else:
                    st.warning("No matching document was found.")

    st.divider()

    st.markdown(
        '<div class="side-label">ABOUT NEXUS</div>',
        unsafe_allow_html=True,
    )

    st.caption("Local AI document intelligence.")

    st.markdown(
        """
        <div class="sidebar-note">
          <b>✦ Privacy-aware workspace</b><br>
          Your language model and vector database run locally.
          Verify important findings against their source documents.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 7. MAIN HEADER
# =========================================================

header_left, header_right = st.columns([5, 1])

with header_left:
    st.markdown(
        """
        <div>
          <div class="topbar-title">Nexus AI</div>
          <div class="topbar-subtitle">
            Business intelligence from your documents
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        """
        <div style="text-align:right;padding-top:5px">
          <span class="local-pill">● LOCAL AI</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# =========================================================
# 8. WELCOME SCREEN
# =========================================================

if not st.session_state.messages:

    st.markdown(
        """
        <div class="welcome">
          <div class="welcome-icon">✦</div>
          <div class="welcome-title">
            What would you like<br>to understand today?
          </div>
          <div class="welcome-text">
            Ask questions about your reports, explore business
            performance, and inspect the evidence behind your answers.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    suggestions = [
        (
            "Revenue analysis",
            "Summarize the key revenue figures.",
        ),
        (
            "Regional performance",
            "Which region generated the highest revenue?",
        ),
        (
            "Business challenges",
            "What are the main challenges in the reports?",
        ),
        (
            "Missing information",
            "What important information is missing from the reports?",
        ),
    ]

    left_col, right_col = st.columns(2)

    for i, (title, question_text) in enumerate(suggestions):
        target_col = left_col if i % 2 == 0 else right_col

        with target_col:
            st.markdown(
                f"""
                <div class="suggestion-card">
                  <div class="suggestion-title">
                    {html.escape(title)}
                  </div>
                  <div class="suggestion-text">
                    {html.escape(question_text)}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                "Explore →",
                key=f"suggestion_{i}",
                use_container_width=True,
            ):
                st.session_state.pending_question = question_text
                st.rerun()


# =========================================================
# 9. CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        citations = message.get("citations", [])

        if citations:
            with st.expander(
                f"View sources and evidence ({len(citations)})"
            ):
                for i, citation in enumerate(citations, start=1):
                    source_label = (
                        f"{citation['source']} · Page/chunk "
                        f"{citation['page']}"
                    )

                    st.markdown(
                        f"**Source {i}: "
                        f"{html.escape(source_label)}**"
                    )

                    st.markdown(
                        '<div class="source-card">'
                        + html.escape(citation["text"])
                        + "</div>",
                        unsafe_allow_html=True,
                    )


# =========================================================
# 10. CHAT INPUT AND RESPONSES
# =========================================================

typed_question = st.chat_input("Message Nexus AI...")

question = typed_question

if st.session_state.pending_question is not None:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        try:
            with st.spinner("Searching your documents..."):
                answer, citations = answer_question(question)

            st.markdown(answer)

            if citations:
                with st.expander(
                    f"View sources and evidence ({len(citations)})"
                ):
                    for i, citation in enumerate(citations, start=1):
                        source_label = (
                            f"{citation['source']} · Page/chunk "
                            f"{citation['page']}"
                        )

                        st.markdown(
                            f"**Source {i}: "
                            f"{html.escape(source_label)}**"
                        )

                        st.markdown(
                            '<div class="source-card">'
                            + html.escape(citation["text"])
                            + "</div>",
                            unsafe_allow_html=True,
                        )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "citations": citations,
            })

        except Exception as error:
            st.error(f"Unable to generate an answer: {error}")
            st.caption(
                "Check that Ollama is running, both models are "
                "installed, and business_rules.py is available."
            )

st.markdown(
    """
    <div class="footer-note">
      NEXUS AI · LOCAL DOCUMENT INTELLIGENCE<br>
      AI-generated answers may be incorrect. Verify important findings.
    </div>
    """,
    unsafe_allow_html=True,
)
