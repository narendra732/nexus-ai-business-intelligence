from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from pypdf import PdfReader

from cloud_ai import generate_answer, using_remote_ai

st.set_page_config(
    page_title="Nexus AI | Report Data Extractor",
    page_icon="📄",
    layout="wide",
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
SALES_FILE = DATA_DIR / "sales_metrics.csv"
REGION_FILE = DATA_DIR / "regional_sales.csv"

MAX_TEXT_PAGES = 5
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_CHARS_PER_PAGE = 8000

st.markdown(
    """
    <style>
    .stApp { background: #f7f8fc; color: #252832; }
    .block-container { max-width: 1200px; padding-top: 2rem; }
    .eyebrow { color: #5b5ce2; text-transform: uppercase; letter-spacing: 1.8px;
               font-size: 10px; font-weight: 700; margin-bottom: 10px; }
    .hero-title { font-size: clamp(28px, 4vw, 38px); font-weight: 800;
                  letter-spacing: -1.2px; line-height: 1.25; }
    .hero-subtitle { color: #7c8392; font-size: 14px; line-height: 1.8;
                     max-width: 760px; margin: 10px 0 25px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="eyebrow">NEXUS AI / DOCUMENT INTELLIGENCE</div>'
    '<div class="hero-title">Report Data Extractor</div>'
    '<div class="hero-subtitle">Extract quarterly and regional revenue from a '
    'business report, review the evidence, and download clean CSV files.</div>',
    unsafe_allow_html=True,
)

if using_remote_ai():
    st.info(
        "Hosted demo: extraction uses the Gemini API. Upload fictional or "
        "sample reports only; do not upload confidential, personal, or client data. "
        "This page exports CSV downloads and does not overwrite shared dashboard files."
    )
else:
    st.info(
        "Local mode uses Ollama when no GEMINI_API_KEY is configured. "
        "This public-demo page exports reviewed CSV downloads and does not overwrite "
        "the dashboard's source files."
    )

def normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def parse_amount(value):
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip().replace(",", "").replace("$", "")
    try:
        amount = float(text)
    except (TypeError, ValueError):
        return None
    if amount < 0 or not pd.notna(amount):
        return None
    return amount


def number_appears_in_text(value, source_text: str) -> bool:
    expected = parse_amount(value)
    if expected is None:
        return False
    candidates = re.findall(r"(?<![A-Za-z])\$?\s*\d[\d,]*(?:\.\d+)?", source_text)
    for candidate in candidates:
        actual = parse_amount(candidate)
        if actual is not None and abs(actual - expected) < 0.000001:
            return True
    return False


def supported_evidence(quote, amount, source_text: str) -> bool:
    quote = str(quote or "").strip()
    if not quote or not source_text:
        return False
    return (
        normalise_text(quote) in normalise_text(source_text)
        and number_appears_in_text(amount, source_text)
    )


def read_pages(uploaded_file):
    raw = uploaded_file.getvalue()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("The file exceeds the 5 MB limit for this public demo.")
    filename = Path(uploaded_file.name).name
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(BytesIO(raw))
        pages = [
            {"page": i, "text": (p.extract_text() or "")}
            for i, p in enumerate(reader.pages, start=1)
            if (p.extract_text() or "").strip()
        ]
    elif suffix == ".txt":
        text = raw.decode("utf-8-sig", errors="replace")
        pages = [{"page": 1, "text": text}] if text.strip() else []
    else:
        raise ValueError("Upload a PDF or TXT file.")
    return filename, pages


def extract_page_metrics(page_text, page_number, source_name):
    prompt = f"""
Extract every explicitly stated quarterly revenue figure and every explicitly
stated regional revenue figure from the source text. Do not calculate missing
values and do not confuse total company revenue with regional revenue.

Return valid JSON with exactly these keys and shapes:
{{
  "quarterly_revenue": [
    {{"quarter": "Q2", "revenue": 520000, "evidence": "exact quote"}}
  ],
  "regional_revenue": [
    {{"region": "North", "revenue": 190000, "evidence": "exact quote"}}
  ]
}}

Rules:
- Include every explicit revenue value in this page, not only the largest value.
- Do not extract costs, percentages, profit margins, customer counts or forecasts.
- Preserve quarter and region labels as stated.
- Evidence must be a verbatim quote from the provided source text.
- Revenue must be a non-negative JSON number without currency symbols or commas.
- Use empty arrays if a category has no explicit values.
- Return JSON only, without Markdown fences.

Source filename: {source_name}
Source page: {page_number}

SOURCE TEXT:
{page_text}
"""
    raw = generate_answer(
        "You are a careful business-data extraction system. Never invent values. "
        "Return only valid JSON that follows the requested schema.",
        prompt,
        local_model="qwen2.5:3b",
        json_mode=True,
    )
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("The model did not return a JSON object.")

    quarterly = result.get("quarterly_revenue", [])
    regional = result.get("regional_revenue", [])
    if not isinstance(quarterly, list) or not isinstance(regional, list):
        raise ValueError("The JSON fields must be lists.")

    q_rows, r_rows = [], []
    for item in quarterly:
        if not isinstance(item, dict):
            continue
        quarter = str(item.get("quarter", "")).strip()
        amount = parse_amount(item.get("revenue"))
        evidence = str(item.get("evidence", "")).strip()
        if not quarter or amount is None:
            continue
        q_rows.append({
            "Quarter": quarter,
            "Revenue_USD": amount,
            "Source_File": source_name,
            "Source_Page": page_number,
            "Evidence": evidence,
            "Grounded": supported_evidence(evidence, amount, page_text),
        })

    for item in regional:
        if not isinstance(item, dict):
            continue
        region = str(item.get("region", "")).strip()
        amount = parse_amount(item.get("revenue"))
        evidence = str(item.get("evidence", "")).strip()
        if not region or amount is None:
            continue
        r_rows.append({
            "Region": region,
            "Revenue_USD": amount,
            "Source_File": source_name,
            "Source_Page": page_number,
            "Evidence": evidence,
            "Grounded": supported_evidence(evidence, amount, page_text),
        })
    return q_rows, r_rows


def validate_export_table(frame: pd.DataFrame, label_column: str) -> str | None:
    if frame.empty:
        return None
    required = {
        label_column, "Revenue_USD", "Source_File",
        "Source_Page", "Evidence", "Grounded"
    }
    if not required.issubset(frame.columns):
        return "The reviewed table is missing required columns."
    labels = frame[label_column].astype("string").str.strip()
    if labels.isna().any() or labels.eq("").any():
        return f"Some {label_column} labels are blank."
    if labels.str.casefold().duplicated().any():
        return f"Duplicate {label_column.lower()} labels must be resolved."
    amounts = pd.to_numeric(frame["Revenue_USD"], errors="coerce")
    if amounts.isna().any() or (amounts < 0).any():
        return "Revenue values must be valid, non-negative numbers."
    if not frame["Grounded"].fillna(False).astype(bool).all():
        return "Every row must pass the source-text check before saving."
    if frame["Evidence"].astype(str).str.strip().eq("").any():
        return "Every row must have supporting evidence."
    return None


def save_local_dashboard_csvs(quarterly: pd.DataFrame, regional: pd.DataFrame):
    """Safely update local CSVs, backing up originals and rolling back on failure."""
    outputs = {}
    for frame, label, path, columns in [
        (quarterly, "Quarter", SALES_FILE,
         ["Quarter", "Revenue_USD", "Source_File", "Source_Page", "Evidence"]),
        (regional, "Region", REGION_FILE,
         ["Region", "Revenue_USD", "Source_File", "Source_Page", "Evidence"]),
    ]:
        if frame.empty:
            continue
        issue = validate_export_table(frame, label)
        if issue:
            raise ValueError(issue)
        outputs[path] = frame[columns].copy()

    if not outputs:
        raise ValueError("There are no extracted rows to save.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    staged, backups, replaced = {}, {}, []
    try:
        for path, frame in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(f".{path.stem}_{stamp}.tmp")
            frame.to_csv(temp, index=False, encoding="utf-8")
            pd.read_csv(temp)
            staged[path] = temp

        for path in outputs:
            if path.exists():
                backup = path.with_name(
                    f"{path.stem}_backup_{stamp}{path.suffix}"
                )
                shutil.copy2(path, backup)
                backups[path] = backup
            else:
                backups[path] = None

        for path, temp in staged.items():
            os.replace(temp, path)
            replaced.append(path)
    except Exception:
        for path in reversed(replaced):
            backup = backups.get(path)
            try:
                if backup and backup.exists():
                    shutil.copy2(backup, path)
                elif path.exists():
                    path.unlink()
            except OSError:
                pass
        raise
    finally:
        for temp in staged.values():
            try:
                if temp.exists():
                    temp.unlink()
            except OSError:
                pass
    return [str(path) for path in outputs], [
        str(path) for path in backups.values() if path is not None
    ]


if "extractor_data" not in st.session_state:
    st.session_state["extractor_data"] = None

uploaded = st.file_uploader("Upload a PDF or TXT business report", type=["pdf", "txt"])
if st.button("Extract revenue metrics", type="primary", use_container_width=True):
    if uploaded is None:
        st.warning("Upload a report first.")
    else:
        try:
            filename, pages = read_pages(uploaded)
            if not pages:
                st.error("No extractable text was found. Scanned PDFs may require OCR.")
            else:
                if len(pages) > MAX_TEXT_PAGES:
                    st.warning(
                        f"To keep the public demo responsive, only the first "
                        f"{MAX_TEXT_PAGES} text-bearing pages will be processed."
                    )
                all_q, all_r = [], []
                progress = st.progress(0)
                process_pages = pages[:MAX_TEXT_PAGES]
                for i, page in enumerate(process_pages):
                    text = page["text"][:MAX_CHARS_PER_PAGE]
                    q_rows, r_rows = extract_page_metrics(text, page["page"], filename)
                    all_q.extend(q_rows)
                    all_r.extend(r_rows)
                    progress.progress((i + 1) / len(process_pages))
                st.session_state["extractor_data"] = {
                    "source": filename,
                    "quarterly": all_q,
                    "regional": all_r,
                    "page_count": len(process_pages),
                }
                st.success(f"Processed {len(process_pages)} text-bearing page(s). Review each result below.")
        except Exception as exc:
            st.error(f"Could not extract the report: {exc}")

result = st.session_state.get("extractor_data")
if result:
    st.caption(f"Source: {result['source']} · Pages processed: {result['page_count']}")
    quarterly = pd.DataFrame(result["quarterly"])
    regional = pd.DataFrame(result["regional"])

    st.subheader("1. Review quarterly revenue")
    if not quarterly.empty:
        edited_quarterly = st.data_editor(
            quarterly,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            disabled=["Source_File", "Source_Page", "Evidence", "Grounded"],
            key=f"cloud_quarterly_{result['source']}",
            column_config={
                "Revenue_USD": st.column_config.NumberColumn("Revenue (USD)", min_value=0.0, format="$%.2f"),
                "Grounded": st.column_config.CheckboxColumn("Source text check"),
            },
        )
        if (~edited_quarterly["Grounded"].fillna(False).astype(bool)).any():
            st.warning("Some rows failed the source-text check. Verify those rows manually; do not rely on them without review.")
        st.download_button(
            "Download quarterly revenue CSV",
            edited_quarterly.to_csv(index=False).encode("utf-8"),
            file_name="extracted_quarterly_revenue.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        edited_quarterly = pd.DataFrame()
        st.info("No quarterly revenue figures were extracted.")

    st.subheader("2. Review regional revenue")
    if not regional.empty:
        edited_regional = st.data_editor(
            regional,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            disabled=["Source_File", "Source_Page", "Evidence", "Grounded"],
            key=f"cloud_regional_{result['source']}",
            column_config={
                "Revenue_USD": st.column_config.NumberColumn("Revenue (USD)", min_value=0.0, format="$%.2f"),
                "Grounded": st.column_config.CheckboxColumn("Source text check"),
            },
        )
        if (~edited_regional["Grounded"].fillna(False).astype(bool)).any():
            st.warning("Some rows failed the source-text check. Verify those rows manually; do not rely on them without review.")
        st.download_button(
            "Download regional revenue CSV",
            edited_regional.to_csv(index=False).encode("utf-8"),
            file_name="extracted_regional_revenue.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        edited_regional = pd.DataFrame()
        st.info("No regional revenue figures were extracted.")

    if using_remote_ai():
        st.caption(
            "This public demo keeps extraction results in the active session and offers "
            "CSV downloads. It does not overwrite the shared dashboard data files. "
            "A text match is only a basic grounding check; review figures against the source."
        )
    else:
        st.divider()
        st.subheader("3. Approve local dashboard updates")
        st.write(
            "Local mode can replace the dashboard CSVs after a backup is created. "
            "Only rows that pass the source-text check may be saved."
        )
        approved = st.checkbox(
            "I reviewed the extracted labels, amounts, and source evidence.",
            key=f"local_save_confirm_{result['source']}",
        )
        if st.button(
            "Approve and update local dashboard CSVs",
            type="primary",
            disabled=not approved,
            use_container_width=True,
        ):
            try:
                written, backups = save_local_dashboard_csvs(
                    edited_quarterly, edited_regional
                )
                st.success("Local dashboard data updated.")
                st.write("Updated files:")
                for filename in written:
                    st.code(filename, language=None)
                if backups:
                    st.write("Backups:")
                    for filename in backups:
                        st.code(filename, language=None)
            except Exception as exc:
                st.error(f"Could not update local dashboard files: {exc}")

st.markdown(
    '<div style="text-align:center;padding-top:30px;color:#9298a6;font-size:11px">'
    'NEXUS AI · REPORT DATA EXTRACTOR<br>'
    'AI-extracted values require human verification before use.</div>',
    unsafe_allow_html=True,
)
