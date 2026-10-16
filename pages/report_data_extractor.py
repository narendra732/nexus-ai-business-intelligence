
from pathlib import Path
from io import BytesIO
from datetime import datetime
import hashlib
import json
import math
import os
import re
import shutil

import pandas as pd
import ollama
import streamlit as st
from pypdf import PdfReader


# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Nexus AI | Report Data Extractor",
    page_icon="📄",
    layout="wide",
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

SALES_FILE = DATA_DIR / "sales_metrics.csv"
REGION_FILE = DATA_DIR / "regional_sales.csv"

CHAT_MODEL = "qwen2.5:3b"

MAX_TEXT_PAGES = 20
MAX_CHARS_PER_PAGE = 9000


# =========================================================
# 2. DESIGN
# =========================================================

st.markdown(
    """
    <style>
    @import url(
      'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap'
    );

    .stApp {
        background: #f7f8fc;
        color: #252832;
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        letter-spacing: -0.7px;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .eyebrow {
        color: #5b5ce2;
        text-transform: uppercase;
        letter-spacing: 1.8px;
        font-size: 10px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .hero-title {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(28px, 4vw, 38px);
        font-weight: 800;
        letter-spacing: -1.4px;
        line-height: 1.25;
        color: #252832;
    }

    .hero-subtitle {
        color: #7c8392;
        font-size: 14px;
        line-height: 1.8;
        max-width: 760px;
        margin: 10px 0 25px;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 40px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #ffffff;
        border: 1px dashed #c9cddd;
        border-radius: 12px;
    }

    [data-testid="stExpander"] {
        border: 1px solid #e7eaf1;
        border-radius: 12px;
        background: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. SESSION STATE
# =========================================================

if "extractor_results" not in st.session_state:
    st.session_state.extractor_results = None

if "extractor_pages" not in st.session_state:
    st.session_state.extractor_pages = {}


# =========================================================
# 4. READ PDF OR TXT
# =========================================================

def read_uploaded_document(uploaded_file):
    """Extract page text while preserving original page numbers."""

    raw = uploaded_file.getvalue()
    filename = Path(uploaded_file.name).name
    suffix = Path(filename).suffix.lower()

    pages = []

    if suffix == ".pdf":
        reader = PdfReader(BytesIO(raw))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                pages.append({
                    "page": page_number,
                    "text": text,
                })

    elif suffix == ".txt":
        text = raw.decode("utf-8-sig", errors="replace")

        if text.strip():
            pages.append({
                "page": 1,
                "text": text,
            })

    return filename, pages


# =========================================================
# 5. NUMERIC PARSING AND EVIDENCE VALIDATION
# =========================================================

def parse_amount(value):
    """Parse a non-negative, finite revenue amount."""

    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        amount = float(value)

    else:
        text = str(value).strip()
        text = text.replace(",", "").replace("$", "").strip()

        try:
            amount = float(text)
        except ValueError:
            return None

    if not math.isfinite(amount) or amount < 0:
        return None

    return amount


def normalise_text(text):
    return re.sub(
        r"\s+",
        " ",
        str(text),
    ).strip().casefold()


def number_appears_in_text(value, source_text):
    """Check whether the exact amount appears on the source page."""

    expected = parse_amount(value)

    if expected is None:
        return False

    candidates = re.findall(
        r"(?<![A-Za-z])\$?\s*\d[\d,]*(?:\.\d+)?",
        source_text,
    )

    for candidate in candidates:
        actual = parse_amount(candidate)

        if actual is not None and math.isclose(
            actual,
            expected,
            rel_tol=0,
            abs_tol=0.000001,
        ):
            return True

    return False


def evidence_is_supported(evidence, value, page_text):
    """
    Basic grounding check:
    the quote and numeric amount must occur on the same source page.
    """

    quote = str(evidence or "").strip()

    if not quote or not page_text:
        return False

    quote_found = (
        normalise_text(quote) in normalise_text(page_text)
    )

    amount_found = number_appears_in_text(
        value,
        page_text,
    )

    return quote_found and amount_found


def recheck_grounding(df):
    """
    Recheck edited amounts against the page from which they
    were originally extracted.
    """

    if df.empty:
        return df

    checked = df.copy()

    result = st.session_state.extractor_results
    source_pages = st.session_state.extractor_pages

    statuses = []

    for _, row in checked.iterrows():
        source_name = str(row.get("Source_File", ""))

        page_value = row.get("Source_Page")

        page_key = ""

        if pd.notna(page_value):
            try:
                page_key = str(int(page_value))
            except (TypeError, ValueError, OverflowError):
                page_key = ""

        source_text = source_pages.get(page_key, "")

        # Only verify citations against the currently loaded
        # source document, not a different uploaded filename.
        correct_source = (
            result is not None
            and source_name == result["source"]
        )

        supported = (
            correct_source
            and evidence_is_supported(
                row.get("Evidence", ""),
                row.get("Revenue_USD"),
                source_text,
            )
        )

        statuses.append(bool(supported))

    checked["Grounded"] = statuses

    return checked


# =========================================================
# 6. EXTRACT BUSINESS METRICS WITH LOCAL OLLAMA
# =========================================================

def extract_page_metrics(page_text, page_number, source_name):
    """
    Extract all explicitly stated quarterly and regional revenue.
    This function does not calculate missing figures.
    """

    prompt = f"""
You are a meticulous business-data extraction system.

Scan the ENTIRE source text. Extract EVERY explicitly reported
quarterly revenue and EVERY explicitly reported regional revenue.

Return valid JSON with exactly these keys:

{{
  "quarterly_revenue": [
    {{
      "quarter": "Q2",
      "revenue": 520000,
      "evidence": "Exact supporting quote"
    }}
  ],
  "regional_revenue": [
    {{
      "region": "North",
      "revenue": 190000,
      "evidence": "Exact supporting quote"
    }}
  ]
}}

Rules:
- Extract every quarter with an explicit revenue figure.
- Extract every region with an explicit revenue figure.
- A comparison sentence may contain multiple quarter values.
  Extract each value separately with its correct quarter.
- Do not return only the highest-revenue region.
- Do not confuse total company revenue with regional revenue.
- Do not calculate missing values.
- Do not extract costs, percentages, profit margins, or customer counts.
- Preserve the quarter labels as reported in the source.
- Evidence must be an exact quote from the supplied source.
- Revenue must be a JSON number without currency symbols or commas.
- Return an empty list when the category has no explicit values.
- Return valid JSON only, without Markdown.
- Check the full text before returning the output.

Source filename: {source_name}
Source page: {page_number}

SOURCE TEXT:
{page_text}
"""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract all explicitly stated business revenue "
                    "figures. Be exhaustive and never invent values. "
                    "Return valid JSON."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        format="json",
        options={"temperature": 0},
    )

    result = json.loads(
        response["message"]["content"]
    )

    if not isinstance(result, dict):
        raise ValueError("Unexpected JSON response from the model.")

    quarterly = result.get("quarterly_revenue", [])
    regional = result.get("regional_revenue", [])

    if not isinstance(quarterly, list):
        raise ValueError("quarterly_revenue must be a list.")

    if not isinstance(regional, list):
        raise ValueError("regional_revenue must be a list.")

    output_quarterly = []
    output_regional = []

    for item in quarterly:
        if not isinstance(item, dict):
            continue

        amount = parse_amount(item.get("revenue"))
        quarter = str(item.get("quarter", "")).strip()
        evidence = str(item.get("evidence", "")).strip()

        if not quarter or amount is None:
            continue

        output_quarterly.append({
            "Quarter": quarter,
            "Revenue_USD": amount,
            "Source_File": source_name,
            "Source_Page": page_number,
            "Evidence": evidence,
            "Grounded": evidence_is_supported(
                evidence,
                amount,
                page_text,
            ),
        })

    for item in regional:
        if not isinstance(item, dict):
            continue

        amount = parse_amount(item.get("revenue"))
        region = str(item.get("region", "")).strip()
        evidence = str(item.get("evidence", "")).strip()

        if not region or amount is None:
            continue

        output_regional.append({
            "Region": region,
            "Revenue_USD": amount,
            "Source_File": source_name,
            "Source_Page": page_number,
            "Evidence": evidence,
            "Grounded": evidence_is_supported(
                evidence,
                amount,
                page_text,
            ),
        })

    return output_quarterly, output_regional


# =========================================================
# 7. DATA VALIDATION
# =========================================================

def validate_table(df, label_column):
    """Reject invalid or duplicated labels and revenue values."""

    if df.empty:
        return "There are no rows to save."

    required = {
        label_column,
        "Revenue_USD",
        "Source_File",
        "Source_Page",
        "Evidence",
        "Grounded",
    }

    if not required.issubset(df.columns):
        return "The table is missing required columns."

    labels = df[label_column].astype("string").str.strip()

    if labels.isna().any() or labels.eq("").any():
        return f"Some {label_column.lower()} labels are missing."

    if labels.str.casefold().duplicated().any():
        return (
            f"Duplicate {label_column.lower()} labels exist. "
            "Resolve the duplicates before saving."
        )

    amounts = pd.to_numeric(
        df["Revenue_USD"],
        errors="coerce",
    )

    if amounts.isna().any():
        return "Some revenue values are missing or invalid."

    if (amounts < 0).any():
        return "Revenue values cannot be negative."

    if not all(math.isfinite(float(v)) for v in amounts):
        return "Revenue values must be finite numbers."

    if df["Source_File"].isna().any():
        return "Some source filenames are missing."

    if df["Source_Page"].isna().any():
        return "Some source page numbers are missing."

    if df["Evidence"].isna().any():
        return "Some source evidence is missing."

    if df["Evidence"].astype(str).str.strip().eq("").any():
        return "Some source evidence is empty."

    if (~df["Grounded"].fillna(False).astype(bool)).any():
        return "One or more rows failed evidence verification."

    return None


# =========================================================
# 8. BACKUP AND SAFER CSV WRITING
# =========================================================

def backup_existing_file(path):
    """Make a timestamped copy before replacing existing data."""

    if not path.exists():
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    backup_path = path.with_name(
        f"{path.stem}_backup_{timestamp}{path.suffix}"
    )

    shutil.copy2(path, backup_path)

    return backup_path


def write_dashboard_csvs(quarterly_df, regional_df):
    """
    Validate both tables before changing files.

    Stage the new CSVs first. Create backups, then replace the
    existing files. Restore backups if a replacement fails.
    """

    outputs = {}

    if not quarterly_df.empty:
        error = validate_table(
            quarterly_df,
            "Quarter",
        )

        if error:
            raise ValueError(
                f"Quarterly revenue validation failed: {error}"
            )

        outputs[SALES_FILE] = quarterly_df[
            [
                "Quarter",
                "Revenue_USD",
                "Source_File",
                "Source_Page",
                "Evidence",
            ]
        ].copy()

    if not regional_df.empty:
        error = validate_table(
            regional_df,
            "Region",
        )

        if error:
            raise ValueError(
                f"Regional revenue validation failed: {error}"
            )

        outputs[REGION_FILE] = regional_df[
            [
                "Region",
                "Revenue_USD",
                "Source_File",
                "Source_Page",
                "Evidence",
            ]
        ].copy()

    if not outputs:
        raise ValueError("No extracted rows are available to save.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    temp_paths = {}
    backups = {}
    replaced_paths = []

    try:
        # Stage every output before touching existing CSVs.
        for path, frame in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)

            temp_path = path.with_name(
                f".{path.stem}_{timestamp}.tmp"
            )

            frame.to_csv(
                temp_path,
                index=False,
                encoding="utf-8",
            )

            # Confirm the staged file is readable.
            pd.read_csv(temp_path)

            temp_paths[path] = temp_path

        # Back up every existing destination before replacement.
        for path in outputs:
            backups[path] = backup_existing_file(path)

        # Replace staged files.
        for path, temp_path in temp_paths.items():
            os.replace(temp_path, path)
            replaced_paths.append(path)

    except Exception:
        # Best-effort rollback if a replacement has already occurred.
        for path in reversed(replaced_paths):
            backup = backups.get(path)

            try:
                if backup and Path(backup).exists():
                    shutil.copy2(backup, path)
                elif path.exists():
                    path.unlink()
            except OSError:
                pass

        raise

    finally:
        for temp_path in temp_paths.values():
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass

    written_files = [str(path) for path in outputs]

    backup_files = [
        str(backup)
        for backup in backups.values()
        if backup is not None
    ]

    return written_files, backup_files


# =========================================================
# 9. PAGE HEADER
# =========================================================

st.markdown(
    '<div class="eyebrow">NEXUS AI / DOCUMENT INTELLIGENCE</div>'
    '<div class="hero-title">Report Data Extractor</div>'
    '<div class="hero-subtitle">'
    'Turn business reports into structured, reviewable revenue data. '
    'Review the values and original evidence before updating dashboard files.'
    '</div>',
    unsafe_allow_html=True,
)

st.info(
    "Extraction runs through your local Ollama model. "
    "This version supports PDF and TXT files with extractable text. "
    "Scanned PDFs may require OCR."
)


# =========================================================
# 10. UPLOAD AND EXTRACTION
# =========================================================

uploaded = st.file_uploader(
    "Upload a PDF or TXT business report",
    type=["pdf", "txt"],
    accept_multiple_files=False,
)

if st.button(
    "Extract revenue metrics",
    type="primary",
    use_container_width=True,
):
    if uploaded is None:
        st.warning("Upload a report first.")

    else:
        try:
            source_name, pages = read_uploaded_document(
                uploaded
            )

            if not pages:
                st.error(
                    "No extractable text was found. "
                    "The PDF may be scanned or image-only."
                )

            else:
                if len(pages) > MAX_TEXT_PAGES:
                    st.warning(
                        f"The document has more than {MAX_TEXT_PAGES} "
                        "text-bearing pages. Only the first "
                        f"{MAX_TEXT_PAGES} will be processed."
                    )

                pages_to_process = pages[:MAX_TEXT_PAGES]

                all_quarterly = []
                all_regional = []
                source_page_texts = {}

                progress = st.progress(0)
                status = st.empty()

                for i, page in enumerate(pages_to_process):
                    page_number = page["page"]
                    page_text = page["text"][:MAX_CHARS_PER_PAGE]

                    source_page_texts[str(page_number)] = page_text

                    status.write(
                        f"Extracting page {page_number} "
                        f"({i + 1}/{len(pages_to_process)})..."
                    )

                    try:
                        quarterly, regional = extract_page_metrics(
                            page_text=page_text,
                            page_number=page_number,
                            source_name=source_name,
                        )

                        all_quarterly.extend(quarterly)
                        all_regional.extend(regional)

                    except Exception as error:
                        st.warning(
                            f"Page {page_number} could not be "
                            f"processed: {error}"
                        )

                    progress.progress(
                        (i + 1) / len(pages_to_process)
                    )

                st.session_state.extractor_results = {
                    "source": source_name,
                    "quarterly": all_quarterly,
                    "regional": all_regional,
                }

                st.session_state.extractor_pages = source_page_texts

                status.empty()

                st.success(
                    f"Processed {len(pages_to_process)} "
                    "text-bearing page(s). Review all extracted "
                    "values before approving them."
                )

        except Exception as error:
            st.error(f"Unable to process the report: {error}")


# =========================================================
# 11. DISPLAY AND EDIT EXTRACTION RESULTS
# =========================================================

result = st.session_state.extractor_results

if result is not None:

    quarterly_rows = result["quarterly"]
    regional_rows = result["regional"]

    st.divider()
    st.subheader("1. Review quarterly revenue")

    if quarterly_rows:
        quarterly_df = pd.DataFrame(quarterly_rows)

        edited_quarterly = st.data_editor(
            quarterly_df,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            key=f"quarterly_{result['source']}",
            disabled=[
                "Source_File",
                "Source_Page",
                "Evidence",
                "Grounded",
            ],
            column_config={
                "Revenue_USD": st.column_config.NumberColumn(
                    "Revenue (USD)",
                    min_value=0.0,
                    format="$%.2f",
                ),
                "Grounded": st.column_config.CheckboxColumn(
                    "Original evidence check",
                    help="Initial check performed during extraction.",
                ),
            },
        )

        edited_quarterly = recheck_grounding(
            edited_quarterly
        )

        quarterly_failures = int(
            (
                ~edited_quarterly["Grounded"]
                .fillna(False)
                .astype(bool)
            ).sum()
        )

        if quarterly_failures:
            st.warning(
                f"{quarterly_failures} quarterly row(s) currently "
                "fail evidence verification. Check the edited "
                "values against the report."
            )
        else:
            st.success(
                "All quarterly rows currently pass the basic "
                "source-text check."
            )

        st.caption(
            "The original source filename, page and evidence are "
            "preserved. Edited figures are checked again."
        )

    else:
        edited_quarterly = pd.DataFrame()
        st.info("No quarterly revenue figures were extracted.")

    st.subheader("2. Review regional revenue")

    if regional_rows:
        regional_df = pd.DataFrame(regional_rows)

        edited_regional = st.data_editor(
            regional_df,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            key=f"regional_{result['source']}",
            disabled=[
                "Source_File",
                "Source_Page",
                "Evidence",
                "Grounded",
            ],
            column_config={
                "Revenue_USD": st.column_config.NumberColumn(
                    "Revenue (USD)",
                    min_value=0.0,
                    format="$%.2f",
                ),
                "Grounded": st.column_config.CheckboxColumn(
                    "Original evidence check",
                    help="Initial check performed during extraction.",
                ),
            },
        )

        edited_regional = recheck_grounding(
            edited_regional
        )

        regional_failures = int(
            (
                ~edited_regional["Grounded"]
                .fillna(False)
                .astype(bool)
            ).sum()
        )

        if regional_failures:
            st.warning(
                f"{regional_failures} regional row(s) currently "
                "fail evidence verification. Check the edited "
                "values against the report."
            )
        else:
            st.success(
                "All regional rows currently pass the basic "
                "source-text check."
            )

        st.caption(
            "Review every region, not just the highest-revenue region. "
            "Remove accidental duplicate rows in the editor."
        )

    else:
        edited_regional = pd.DataFrame()
        st.info("No regional revenue figures were extracted.")


    # =====================================================
    # 12. RECONCILIATION
    # =====================================================

    st.divider()
    st.subheader("3. Revenue reconciliation")

    reconciliation_ok = True

    if (
        not edited_quarterly.empty
        and not edited_regional.empty
    ):
        quarter_options = [
            str(value).strip()
            for value in edited_quarterly["Quarter"].tolist()
            if pd.notna(value) and str(value).strip()
        ]

        if quarter_options:
            selected_quarter = st.selectbox(
                "Select a quarter to compare against the regional total",
                options=quarter_options,
                key=f"reconcile_quarter_{result['source']}",
            )

            selected_rows = edited_quarterly[
                edited_quarterly["Quarter"].astype(str).str.strip()
                == selected_quarter
            ]

            # Duplicate labels are independently rejected at save time.
            # Use the first row here so the user sees a result.
            if selected_rows.empty:
                st.warning(
                    "No revenue value is available for this quarter."
                )
                reconciliation_ok = False

            else:
                selected_amounts = pd.to_numeric(
                    selected_rows["Revenue_USD"],
                    errors="coerce",
                )

                regional_amounts = pd.to_numeric(
                    edited_regional["Revenue_USD"],
                    errors="coerce",
                )

                if (
                    selected_amounts.isna().any()
                    or regional_amounts.isna().any()
                ):
                    st.warning(
                        "Reconciliation cannot be calculated until "
                        "all revenue values are numeric."
                    )
                    reconciliation_ok = False

                else:
                    quarter_total = float(
                        selected_amounts.iloc[0]
                    )

                    regional_total = float(
                        regional_amounts.sum()
                    )

                    difference = regional_total - quarter_total

                    match = math.isclose(
                        regional_total,
                        quarter_total,
                        rel_tol=0,
                        abs_tol=0.01,
                    )

                    left, middle, right = st.columns(3)

                    left.metric(
                        "Selected quarter",
                        f"${quarter_total:,.2f}",
                    )

                    middle.metric(
                        "Regional total",
                        f"${regional_total:,.2f}",
                    )

                    right.metric(
                        "Difference",
                        f"${difference:,.2f}",
                    )

                    if match:
                        st.success(
                            "Regional revenue reconciles with "
                            "the selected quarter."
                        )

                    else:
                        st.warning(
                            "These totals do not match. Check the "
                            "periods, currencies, missing regions, "
                            "and duplicate entries before saving."
                        )

                        reconciliation_ok = st.checkbox(
                            "I investigated the difference and "
                            "confirmed these figures are appropriate "
                            "to save.",
                            key=f"reconcile_confirm_{result['source']}",
                        )

        else:
            st.warning(
                "No valid quarter labels are available for reconciliation."
            )
            reconciliation_ok = False

    else:
        st.info(
            "Reconciliation needs both quarterly and regional data. "
            "Only the available data category can be reviewed."
        )


    # =====================================================
    # 13. SAVE GATE
    # =====================================================

    st.divider()
    st.subheader("4. Approve dashboard updates")

    st.write(
        "The approved CSV files will be used by your Analytics "
        "Dashboard. When a category is saved, its existing CSV is "
        "replaced with the reviewed rows from this extraction. "
        "A timestamped backup of an existing CSV is made first."
    )

    st.caption(
        "A successful text check does not prove that a number "
        "belongs to the correct business metric. Verify the "
        "source quotes manually."
    )

    reviewed = st.checkbox(
        "I have reviewed the extracted labels, amounts, periods, "
        "and supporting source evidence.",
        key=f"review_confirm_{result['source']}",
    )

    replace_confirmed = st.checkbox(
        "I understand that approved categories replace the "
        "corresponding dashboard CSV files.",
        key=f"replace_confirm_{result['source']}",
    )

    if st.button(
        "Approve and update dashboard CSVs",
        type="primary",
        use_container_width=True,
        disabled=(
            not reviewed
            or not replace_confirmed
            or not reconciliation_ok
        ),
    ):
        try:
            clean_quarterly = edited_quarterly.copy()
            clean_regional = edited_regional.copy()

            # Convert edited revenue entries to numeric values.
            for frame in (clean_quarterly, clean_regional):
                if not frame.empty:
                    frame["Revenue_USD"] = pd.to_numeric(
                        frame["Revenue_USD"],
                        errors="coerce",
                    )

            # Validate both tables before writing either one.
            written_files, backup_files = write_dashboard_csvs(
                clean_quarterly,
                clean_regional,
            )

            st.success(
                "Dashboard data saved successfully."
            )

            st.write("Updated files:")
            for filename in written_files:
                st.code(filename, language=None)

            if backup_files:
                st.write("Previous versions backed up:")
                for backup in backup_files:
                    st.code(backup, language=None)

            st.info(
                "Refresh the Analytics Dashboard to see the updated "
                "data. Source filename, source page and evidence "
                "are also retained in the generated CSVs."
            )

        except Exception as error:
            st.error(
                f"Could not save dashboard data: {error}"
            )


# =========================================================
# 14. FOOTER
# =========================================================

st.markdown(
    """
    <div style="
      text-align:center;
      padding-top:30px;
      color:#9298a6;
      font-size:11px;
      line-height:1.8;">
      NEXUS AI · REPORT DATA EXTRACTOR<br>
      AI-extracted values require human verification before use.
    </div>
    """,
    unsafe_allow_html=True,
)
