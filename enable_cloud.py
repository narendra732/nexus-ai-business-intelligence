"""Safely add a Gemini-backed deployment mode to Nexus AI.

Run from the project root with: python enable_cloud_mode.py
The script backs up files before writing and aborts if expected source code differs.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKUP_DIR = ROOT / ".nexus_cloud_backup"

CLOUD_AI_CODE = r'''"""Provider helpers for local Ollama and hosted Gemini modes.

If GEMINI_API_KEY is set (for example, in Streamlit Community Cloud secrets),
Nexus AI uses Gemini for chat and embeddings. Otherwise, local Ollama is used.
"""
from __future__ import annotations

import os
import uuid
import streamlit as st

GEMINI_CHAT_MODEL = "gemini-3.5-flash-lite"
GEMINI_EMBED_MODEL = "gemini-embedding-001"


def _api_key() -> str:
    """Read the Gemini key from environment variables or Streamlit secrets."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        # Streamlit raises when no secrets file exists during local runs.
        return ""


def cloud_mode_enabled() -> bool:
    """Return True only when a Gemini API key has been configured."""
    return bool(_api_key())


def get_session_id() -> str:
    """Give each hosted browser session its own document/data namespace."""
    if not cloud_mode_enabled():
        return "local"
    if "_nexus_session_id" not in st.session_state:
        st.session_state["_nexus_session_id"] = uuid.uuid4().hex[:16]
    return str(st.session_state["_nexus_session_id"])


@st.cache_resource(show_spinner=False)
def _gemini_client(api_key: str):
    from google import genai
    return genai.Client(api_key=api_key)


@st.cache_resource(show_spinner=False)
def get_cloud_chroma_client():
    """Use an in-memory Chroma client in hosted mode, never a shared disk DB."""
    import chromadb
    return chromadb.Client()


def embed_texts(
    texts: list[str],
    *,
    local_model: str = "nomic-embed-text",
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """Create embeddings with Gemini in cloud mode or Ollama locally."""
    if not texts:
        return []

    key = _api_key()
    if not key:
        import ollama
        response = ollama.embed(model=local_model, input=texts)
        return response["embeddings"]

    from google.genai import types
    client = _gemini_client(key)
    response = client.models.embed_content(
        model=GEMINI_EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    embeddings = response.embeddings or []
    if len(embeddings) != len(texts):
        raise RuntimeError(
            "Gemini returned an unexpected number of embeddings. "
            "Please retry with fewer text chunks."
        )
    return [list(item.values) for item in embeddings]


def generate_text(
    *,
    system_prompt: str,
    user_prompt: str,
    local_model: str = "qwen2.5:3b",
    temperature: float = 0.0,
    response_mime_type: str | None = None,
) -> str:
    """Generate text with Gemini when configured, otherwise use local Ollama."""
    key = _api_key()
    if not key:
        import ollama
        kwargs = {}
        if response_mime_type == "application/json":
            kwargs["format"] = "json"
        response = ollama.chat(
            model=local_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": temperature},
            **kwargs,
        )
        return response["message"]["content"].strip()

    from google.genai import types
    config_kwargs = {"temperature": temperature}
    if response_mime_type:
        config_kwargs["response_mime_type"] = response_mime_type
    config = types.GenerateContentConfig(**config_kwargs)
    prompt = (
        "SYSTEM INSTRUCTIONS\n"
        f"{system_prompt}\n\n"
        "USER INPUT\n"
        f"{user_prompt}"
    )
    response = _gemini_client(key).models.generate_content(
        model=GEMINI_CHAT_MODEL,
        contents=prompt,
        config=config,
    )
    answer = (response.text or "").strip()
    if not answer:
        raise RuntimeError("The AI provider returned an empty response.")
    return answer
'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Could not safely patch {label}: expected one exact match, found {count}. "
            "No project files have been changed."
        )
    return text.replace(old, new, 1)


def regex_once(text: str, pattern: str, replacement: str, label: str) -> str:
    # A callback preserves backslashes in generated Python source strings (for example \n).
    result, count = re.subn(pattern, lambda _match: replacement, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(
            f"Could not safely patch {label}: expected one code block, found {count}. "
            "No project files have been changed."
        )
    return result


def main() -> None:
    app_path = ROOT / "app.py"
    dashboard_path = ROOT / "pages" / "Analytics_Dashboard.py"
    extractor_path = ROOT / "pages" / "report_data_extractor.py"
    readme_path = ROOT / "README.md"

    for path in (app_path, dashboard_path, extractor_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required file not found: {path.relative_to(ROOT)}. "
                "Open the terminal at the Nexus AI project root and run this script again."
            )

    app = app_path.read_text(encoding="utf-8")
    dashboard = dashboard_path.read_text(encoding="utf-8")
    extractor = extractor_path.read_text(encoding="utf-8")
    if "from cloud_ai import" in app or "from cloud_ai import" in extractor:
        raise RuntimeError(
            "Cloud mode appears to have been added already. The script will not apply twice."
        )

    # Main app: imports, provider mode, session-isolated storage and Chroma collection.
    app = replace_once(
        app,
        "import chromadb\nimport ollama\nimport streamlit as st\nfrom pypdf import PdfReader\n\nfrom business_rules import deterministic_business_answer",
        "import chromadb\nimport ollama\nimport streamlit as st\nfrom pypdf import PdfReader\n\nfrom cloud_ai import (\n    cloud_mode_enabled,\n    get_session_id,\n    get_cloud_chroma_client,\n    embed_texts as provider_embed_texts,\n    generate_text,\n)\nfrom business_rules import deterministic_business_answer",
        "app.py imports",
    )

    old_config = '''PROJECT_DIR = Path(__file__).resolve().parent
DB_DIR = str(PROJECT_DIR / "chroma_db")
UPLOAD_DIR = PROJECT_DIR / "data" / "uploaded_reports"

CHAT_MODEL = "qwen2.5:3b"
EMBED_MODEL = "nomic-embed-text"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

client = chromadb.PersistentClient(path=DB_DIR)

collection = client.get_or_create_collection(
    name="business_reports",
    metadata={"hnsw:space": "cosine"},
)'''
    new_config = '''PROJECT_DIR = Path(__file__).resolve().parent
DB_DIR = str(PROJECT_DIR / "chroma_db")
CLOUD_MODE = cloud_mode_enabled()
SESSION_ID = get_session_id()
UPLOAD_DIR = PROJECT_DIR / "data" / "uploaded_reports"

# The hosted demo isolates uploads and Chroma collections by browser session.
# Local Ollama mode keeps the existing local folder and persistent collection.
if CLOUD_MODE:
    UPLOAD_DIR = UPLOAD_DIR / SESSION_ID

CHAT_MODEL = "qwen2.5:3b"
EMBED_MODEL = "nomic-embed-text"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

if CLOUD_MODE:
    client = get_cloud_chroma_client()
    collection_name = f"business_reports_{SESSION_ID}"
else:
    client = chromadb.PersistentClient(path=DB_DIR)
    collection_name = "business_reports"

collection = client.get_or_create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"},
)'''
    # Match the whole configuration structurally instead of requiring exact
    # whitespace or a particular number of blank lines. This accommodates
    # harmless local formatting differences while still requiring the known
    # project/database/collection variables and expected collection metadata.
    config_pattern = r'''(?ms)^PROJECT_DIR\s*=.*?^collection\s*=\s*client\.get_or_create_collection\(\s*\n\s*name\s*=\s*["\']business_reports["\'],\s*\n\s*metadata\s*=\s*\{"hnsw:space":\s*["\']cosine["\']\},\s*\n\)'''
    config_match = re.search(config_pattern, app)
    if config_match is None:
        raise RuntimeError(
            "Could not find the app.py configuration block (PROJECT_DIR through "
            "the business_reports Chroma collection). No project files have been changed."
        )
    old_config_block = config_match.group(0)
    required_tokens = (
        "PROJECT_DIR", "DB_DIR", "UPLOAD_DIR", "CHAT_MODEL", "EMBED_MODEL",
        "PersistentClient", "get_or_create_collection",
    )
    if not all(token in old_config_block for token in required_tokens):
        raise RuntimeError(
            "The app.py configuration uses unexpected variable names. "
            "No project files have been changed."
        )
    app = app[:config_match.start()] + new_config + app[config_match.end():]

    app = replace_once(
        app,
        '''def embed_texts(texts):
    response = ollama.embed(
        model=EMBED_MODEL,
        input=texts,
    )
    return response["embeddings"]''',
        '''def embed_texts(texts, task_type="RETRIEVAL_DOCUMENT"):
    return provider_embed_texts(
        texts,
        local_model=EMBED_MODEL,
        task_type=task_type,
    )''',
        "app.py embedding function",
    )
    app = replace_once(
        app,
        "question_embedding = embed_texts([question])[0]",
        'question_embedding = embed_texts(\n        [question], task_type="RETRIEVAL_QUERY"\n    )[0]',
        "app.py query embedding task",
    )

    app = regex_once(
        app,
        r'''    response = ollama\.chat\(\n        model=CHAT_MODEL,\n        messages=\[\n.*?\n        options=\{"temperature": 0\},\n    \)\n    return response\["message"\]\["content"\]\.strip\(\), citations''',
        '''    answer = generate_text(
        system_prompt="""You are Nexus AI, a careful business analyst.

Answer using only the supplied evidence.
- Never invent facts, numbers, or explanations.
- Compare relevant figures before drawing conclusions.
- If evidence is insufficient, say so clearly.
- Cite source filenames and page/chunk identifiers.
- Distinguish facts from interpretations.
- Be concise and professional.""",
        user_prompt=(
            f"Retrieved evidence:\\n{context}\\n\\n"
            f"Question: {question}"
        ),
        local_model=CHAT_MODEL,
        temperature=0,
    )
    return answer.strip(), citations''',
        "app.py answer generation",
    )
    app = replace_once(
        app,
        '''with header_right:
    st.markdown(
        """
        <div style="text-align:right;padding-top:5px">
          <span class="local-pill">● LOCAL AI</span>
        </div>
        """,
        unsafe_allow_html=True,
    )''',
        '''with header_right:
    backend_label = "CLOUD AI" if CLOUD_MODE else "LOCAL AI"
    st.markdown(
        f"""
        <div style="text-align:right;padding-top:5px">
          <span class="local-pill">● {backend_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )''',
        "app.py dynamic backend label",
    )
    app = replace_once(
        app,
        'st.caption("Local AI document intelligence.")',
        'st.caption("Cloud-hosted Gemini demo." if CLOUD_MODE else "Local AI document intelligence.")',
        "app.py backend caption",
    )
    app = replace_once(
        app,
        '''          <b>✦ Privacy-aware workspace</b><br>
          Your language model and vector database run locally.
          Verify important findings against their source documents.''',
        '''          <b>Evidence-first workspace</b><br>
          The hosted demo isolates documents by browser session. Use only
          sample or non-confidential reports and verify findings against sources.''',
        "app.py privacy note",
    )

    # Dashboard: use per-session files in cloud mode, without mutating sample data.
    dashboard = replace_once(
        dashboard,
        'import plotly.express as px\nimport streamlit as st\n',
        'import plotly.express as px\nimport streamlit as st\n\nfrom cloud_ai import cloud_mode_enabled, get_session_id\n',
        "dashboard imports",
    )
    old_dashboard_paths = '''PROJECT_DIR = Path(__file__).resolve().parent.parent

SALES_FILE = PROJECT_DIR / "data" / "sales_metrics.csv"
REGION_FILE = PROJECT_DIR / "data" / "regional_sales.csv"'''
    new_dashboard_paths = '''PROJECT_DIR = Path(__file__).resolve().parent.parent
BASE_DATA_DIR = PROJECT_DIR / "data"
CLOUD_MODE = cloud_mode_enabled()
SESSION_ID = get_session_id()
SESSION_DATA_DIR = BASE_DATA_DIR / "cloud_sessions" / SESSION_ID

# Each hosted visitor sees their own approved extracts plus the sample data
# for any category they have not extracted. The repository's sample CSVs stay read-only.
base_sales_file = BASE_DATA_DIR / "sales_metrics.csv"
base_region_file = BASE_DATA_DIR / "regional_sales.csv"
if CLOUD_MODE:
    session_sales_file = SESSION_DATA_DIR / "sales_metrics.csv"
    session_region_file = SESSION_DATA_DIR / "regional_sales.csv"
    SALES_FILE = session_sales_file if session_sales_file.exists() else base_sales_file
    REGION_FILE = session_region_file if session_region_file.exists() else base_region_file
else:
    SALES_FILE = base_sales_file
    REGION_FILE = base_region_file'''
    dashboard = replace_once(dashboard, old_dashboard_paths, new_dashboard_paths, "dashboard session-specific data paths")

    # Report extractor: Gemini mode, session-specific CSV output, and privacy notice.
    extractor = replace_once(
        extractor,
        'import ollama\nimport streamlit as st\nfrom pypdf import PdfReader\n',
        'import ollama\nimport streamlit as st\nfrom pypdf import PdfReader\n\nfrom cloud_ai import cloud_mode_enabled, get_session_id, generate_text\n',
        "report extractor imports",
    )
    old_extractor_config = '''PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

SALES_FILE = DATA_DIR / "sales_metrics.csv"
REGION_FILE = DATA_DIR / "regional_sales.csv"

CHAT_MODEL = "qwen2.5:3b"'''
    new_extractor_config = '''PROJECT_DIR = Path(__file__).resolve().parent.parent
BASE_DATA_DIR = PROJECT_DIR / "data"
CLOUD_MODE = cloud_mode_enabled()
SESSION_ID = get_session_id()

# Keep hosted users' approved extracts separate from repository sample data.
DATA_DIR = BASE_DATA_DIR / "cloud_sessions" / SESSION_ID if CLOUD_MODE else BASE_DATA_DIR
DATA_DIR.mkdir(parents=True, exist_ok=True)
SALES_FILE = DATA_DIR / "sales_metrics.csv"
REGION_FILE = DATA_DIR / "regional_sales.csv"

CHAT_MODEL = "qwen2.5:3b"'''
    extractor = replace_once(extractor, old_extractor_config, new_extractor_config, "report extractor session output paths")

    extractor = regex_once(
        extractor,
        r'''    response = ollama\.chat\(\n        model=CHAT_MODEL,\n        messages=\[\n.*?\n        format="json",\n        options=\{"temperature": 0\},\n    \)\n    result = json\.loads\(\n        response\["message"\]\["content"\]\n    \)''',
        '''    response_text = generate_text(
        system_prompt=(
            "Extract all explicitly stated business revenue figures. "
            "Be exhaustive and never invent values. Return valid JSON."
        ),
        user_prompt=prompt,
        local_model=CHAT_MODEL,
        temperature=0,
        response_mime_type="application/json",
    )
    result = json.loads(response_text)''',
        "report extractor answer generation",
    )
    extractor = replace_once(
        extractor,
        '''st.info(
    "Extraction runs through your local Ollama model. "
    "This version supports PDF and TXT files with extractable text. "
    "Scanned PDFs may require OCR."
)''',
        '''if CLOUD_MODE:
    st.info(
        "Hosted demo: Gemini processes report text through an external API. "
        "Use only sample or non-confidential documents. Supports PDF/TXT "
        "with extractable text; scanned PDFs may require OCR."
    )
else:
    st.info(
        "Extraction runs through your local Ollama model. "
        "This version supports PDF and TXT files with extractable text. "
        "Scanned PDFs may require OCR."
    )''',
        "report extractor privacy notice",
    )

    # Require every patch to succeed before writing anything.
    cloud_ai_path = ROOT / "cloud_ai.py"
    if cloud_ai_path.exists():
        raise RuntimeError("cloud_ai.py already exists; refusing to overwrite it.")

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(app_path, BACKUP_DIR / "app.py")
    shutil.copy2(dashboard_path, BACKUP_DIR / "Analytics_Dashboard.py")
    shutil.copy2(extractor_path, BACKUP_DIR / "report_data_extractor.py")

    app_path.write_text(app, encoding="utf-8")
    dashboard_path.write_text(dashboard, encoding="utf-8")
    extractor_path.write_text(extractor, encoding="utf-8")
    cloud_ai_path.write_text(CLOUD_AI_CODE, encoding="utf-8")

    # Add required dependencies while preserving an existing requirements file.
    req_path = ROOT / "requirements.txt"
    required_deps = [
        "streamlit>=1.40,<2",
        "chromadb>=0.6,<2",
        "ollama>=0.4,<1",
        "pypdf>=5,<7",
        "pandas>=2,<4",
        "plotly>=5,<7",
        "google-genai>=1,<2",
    ]
    existing = req_path.read_text(encoding="utf-8") if req_path.exists() else ""
    lines = existing.splitlines()
    normalised = {line.strip().lower() for line in lines if line.strip() and not line.strip().startswith("#")}
    for dep in required_deps:
        package = dep.split(">=")[0].lower()
        if not any(line.split("=")[0].split("<")[0].split(">")[0].strip().lower() == package for line in normalised):
            lines.append(dep)
    while lines and not lines[0].strip():
        lines.pop(0)
    req_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    # Make backups and hosted-session outputs non-committable.
    gitignore_path = ROOT / ".gitignore"
    gitignore = gitignore_path.read_text(encoding="utf-8") if gitignore_path.exists() else ""
    for rule in (".nexus_cloud_backup/", "data/cloud_sessions/"):
        if rule not in gitignore.splitlines():
            gitignore = gitignore.rstrip() + "\n" + rule + "\n"
    gitignore_path.write_text(gitignore.lstrip("\n"), encoding="utf-8")

    if readme_path.exists():
        readme = readme_path.read_text(encoding="utf-8")
        readme = readme.replace(
            "Keep the document question-answering workflow local, rather than sending document content to a hosted LLM API.",
            "Local runs use Ollama. When GEMINI_API_KEY is configured, the hosted deployment switches to the Gemini API for embeddings and answer generation.",
        )
        if "## Cloud demo deployment" not in readme:
            readme += '''\n\n## Cloud demo deployment\n\nNexus AI supports two backends. Without `GEMINI_API_KEY`, local runs use Ollama with `qwen2.5:3b` and `nomic-embed-text`. With `GEMINI_API_KEY` configured, the app uses Gemini API models for text embeddings and answer generation.\n\nFor Streamlit Community Cloud, deploy `app.py` from the `main` branch and add this secret in the app's Secrets settings (never commit it to Git):\n\n```toml\nGEMINI_API_KEY = "your-key-from-Google-AI-Studio"\n```\n\nThe hosted demo creates session-specific document collections and upload/output folders. Use synthetic or non-confidential documents only. Gemini API free-tier requests may be used by Google to improve products; see the current Gemini API terms and pricing before submitting any data. The sample dashboard data remains unchanged; approved report extracts are scoped to the current browser session.\n'''
        readme_path.write_text(readme, encoding="utf-8")

    print("Cloud mode has been added successfully.")
    print("Backups were saved to .nexus_cloud_backup/ and are ignored by Git.")
    print("Next: review the changed files, create a Gemini API key, and test locally before pushing.")
    print("Cloud mode activates only when GEMINI_API_KEY is configured; otherwise Ollama remains the default.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nPATCH ABORTED: {exc}")
        raise SystemExit(1)
