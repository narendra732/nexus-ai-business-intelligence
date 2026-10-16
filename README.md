# Nexus AI — RAG Business Intelligence Assistant

Nexus AI is a document-grounded business intelligence prototype. It combines retrieval-augmented generation (RAG), vector search, deterministic business rules, and an interactive sales dashboard so users can explore business reports and inspect the evidence behind answers.

## Features

- Ask natural-language questions about uploaded PDF and TXT reports.
- Split extracted text into overlapping chunks and create embeddings.
- Retrieve relevant passages from ChromaDB and show source filename and page references.
- Use deterministic rules for supported regional-revenue ranking questions.
- Explore quarterly and regional sales using the Analytics Dashboard.
- Extract quarterly and regional revenue figures, check reported figures against source text, review outputs, and download CSVs.
- Use Gemini API for a hosted demo or local Ollama models for local development.

## Architecture

1. **Document ingestion:** PDF text is extracted with `pypdf`; TXT reports are read as UTF-8.
2. **Chunking and embedding:** Report text is divided into overlapping word chunks. Gemini embeddings are used when `GEMINI_API_KEY` is configured; otherwise the original local Ollama embedding model is used.
3. **Vector retrieval:** ChromaDB retrieves relevant chunks using cosine distance.
4. **Answer generation:** The retrieved evidence is passed to a language model with instructions not to invent unsupported facts.
5. **Citations and rule-based analysis:** Answers include source information. Recognized revenue-ranking questions are handled by the separate `business_rules.py` logic.
6. **Analytics and extraction:** The dashboard reads sample CSV datasets; the report extractor provides reviewed CSV downloads. In cloud mode, uploaded documents and vector collections are isolated to the visitor's Streamlit session, and the extractor does not overwrite shared dashboard data.

## Technology Stack

- Python
- Streamlit
- ChromaDB
- Retrieval-Augmented Generation (RAG)
- Google Gemini API and `google-genai` SDK for cloud mode
- Ollama with Qwen and Nomic embeddings for local mode
- Pandas, Plotly, and pypdf

## Run locally with Ollama

1. Install Python 3.11 or 3.12.
2. Install Ollama and download the models:

   ```bash
   ollama pull qwen2.5:3b
   ollama pull nomic-embed-text
   ```

3. Create and activate a virtual environment.
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Make sure `business_rules.py`, `data/sales_metrics.csv`, `data/regional_sales.csv`, and the `pages/` directory are present.
6. Start Ollama, then launch the app from the project root:

   ```bash
   streamlit run app.py
   ```

When `GEMINI_API_KEY` is not configured, Nexus AI uses Ollama locally.

## Configure Gemini cloud mode

The hosted demo uses a Gemini API key stored as a Streamlit secret. Do not put the key in Python source code or commit it to GitHub.

1. Create an API key in [Google AI Studio](https://aistudio.google.com/apikey).
2. In Streamlit Community Cloud, open the app's settings and add this under **Secrets**, replacing the placeholder with the actual key:

   ```toml
   GEMINI_API_KEY = "YOUR_API_KEY"
   ```

3. Keep `.streamlit/secrets.toml` out of Git. For local cloud-mode testing only, this file may contain the same TOML entry; it must never be committed.

The API adapter uses the `gemini-3.8-flash` model for text generation and `gemini-embedding-001` for embeddings by default. Model names can be overridden using `NEXUS_GEMINI_MODEL` and `NEXUS_GEMINI_EMBED_MODEL`.

## Deploy to Streamlit Community Cloud

1. Commit the cloud-ready files in this package to your existing GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
3. Create a new app from `narendra732/nexus-ai-business-intelligence`, branch `main`, entrypoint `app.py`.
4. In **Advanced settings → Secrets**, add `GEMINI_API_KEY` as shown above.
5. Deploy and wait for the build to finish.
6. Test the chat, upload a fictional sample report, inspect source citations, open the Analytics Dashboard, and try the report extractor.
7. Only add the Streamlit URL to your resume after confirming it works in a separate browser session.

## Public-demo limits and privacy

- The public chat is limited to 25 questions per session.
- The public app accepts up to three uploaded reports per indexing batch, limits each report to 100 chunks, and limits uploaded files to 5 MB.
- The report extractor processes up to five text-bearing pages and exports reviewed CSV files. It does not overwrite shared dashboard source files in cloud mode.
- Cloud-mode uploaded content is sent to Google's Gemini API to create embeddings and generate answers. Google's free-tier API data-use terms may permit submitted data to be used to improve its products. Use synthetic or fictional examples only. Never upload confidential, personal, regulated, or client data.
- ChromaDB is in-memory and session-scoped in cloud mode. Uploaded documents are not durable storage and may be lost when the service restarts.
- AI extraction and text-match checks can be wrong. Validate figures against the original source before relying on them.

## Evaluation

The repository includes rule-based answer checks. Run them from the project root:

```bash
python evaluate_answers.py
```

The evaluator is a lightweight regression check for the sample questions and rule-based business analysis. It is not a comprehensive benchmark of answer quality.

## Limitations

- PDF ingestion requires extractable text; scanned/image-only PDFs need OCR, which is not included.
- Deterministic revenue ranking supports recognized patterns in the sample data and should not be treated as a general-purpose financial reconciliation engine.
- The public cloud demo uses an external hosted API and is not intended for sensitive company data.
- Session-scoped in-memory vector storage is suitable for a prototype, not a persistent multi-user production system.

## Example dataset

The repository's included revenue datasets are sample/fictional data intended for demonstration. They do not represent verified real-world company performance.

## Repository layout

```text
.
├── app.py
├── cloud_ai.py
├── business_rules.py
├── rag_pipeline.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── pages/
│   ├── Analytics_Dashboard.py
│   └── report_data_extractor.py
└── data/
    ├── sales_metrics.csv
    ├── regional_sales.csv
    └── sample_reports/
```
