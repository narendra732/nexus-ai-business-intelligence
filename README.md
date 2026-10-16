# Nexus AI — Business Intelligence Assistant

Nexus AI is a local-first business intelligence project that combines retrieval-augmented generation (RAG) with practical analytics workflows. I built it to explore a useful question for analysts: how can people ask questions about business reports in plain English while still being able to trace answers back to the source data?

The application lets a user upload PDF or text reports, retrieve relevant passages, and ask questions about the information they contain. It also includes a small sales dashboard and a report-data extraction workflow, so the project covers more than a basic document chatbot.

This is a portfolio project and a learning prototype. It is designed to demonstrate the workflow and engineering decisions; it is not presented as a production-ready enterprise system.

## What it does

### 1. Ask questions about business documents

- Upload PDF and TXT documents through the Streamlit interface.
- Create embeddings locally with Ollama and `nomic-embed-text`.
- Store and search document chunks with ChromaDB.
- Use the local `qwen2.5:3b` model to draft responses using retrieved context.
- Display source references so users can inspect the material behind an answer.
- Keep the document question-answering workflow local, rather than sending document content to a hosted LLM API.

Answer quality depends on the content and quality of the uploaded documents, retrieval settings, and local model. A citation is a way to verify the source—not a guarantee that every generated statement is correct.

### 2. Apply deterministic business rules

Generative models can misread numbers or confidently answer questions that the source does not support. For a limited set of regional-revenue questions, `business_rules.py` uses explicit business logic instead of relying only on generated text. It can compare supported regional revenue figures and avoid returning a ranking when the figures conflict.

This is an intentional design choice: where a question has a clear numerical answer and the data can be parsed reliably, deterministic logic is preferable to asking a language model to do arithmetic from prose. The current rules cover specific supported patterns; they are not a general-purpose financial calculation engine.

### 3. Explore sales metrics in a dashboard

The Streamlit analytics dashboard reads the project's CSV datasets and presents summary views of quarterly and regional sales metrics. The included sample data is for demonstration and should not be interpreted as real company performance.

### 4. Extract and review report data

The report-data extractor helps turn PDF or text report content into structured revenue data. The workflow includes model-assisted extraction, grounding checks, reconciliation, human review, and backup handling before data is written to CSV. Human review remains important: extracted values should be checked against the original report before they are used in a business decision.

## Project structure

```text
Rag Business Analyst/
├── app.py                         # Main Streamlit RAG application
├── business_rules.py              # Deterministic business-answer logic
├── rag_pipeline.py                # RAG pipeline utilities
├── evaluate.py                    # Evaluation utilities
├── evaluate_answers.py             # Answer checks for selected questions
├── sample_report.txt              # Example report
├── data/
│   ├── regional_sales.csv          # Example regional sales data
│   ├── sales_metrics.csv           # Example sales metrics
│   └── sample_reports/
│       └── sample_report.txt       # Example source report
├── pages/
│   ├── Analytics_Dashboard.py      # Sales analytics dashboard
│   └── report_data_extractor.py    # Report extraction workflow
├── chroma_db/                      # Local vector database; generated at runtime
└── .gitignore
```

The exact filenames may differ slightly as the project evolves. The `chroma_db/` directory and local Python environment are intentionally excluded from version control because they can be recreated locally.

## Tech stack

- **Python** for application logic and data processing
- **Streamlit** for the user interface and multi-page app
- **Ollama** to run language and embedding models locally
- **Qwen 2.5 3B** (`qwen2.5:3b`) for local answer generation
- **Nomic Embed Text** (`nomic-embed-text`) for embeddings
- **ChromaDB** for vector storage and similarity retrieval
- **pypdf** for reading PDF files
- **Pandas** for tabular data handling
- **Plotly** for dashboard charts, where used

## Run it locally

These instructions are for Windows and PowerShell, which is the environment used during development.

### 1. Install the prerequisites

Install Python (a version compatible with the dependencies), Git, and [Ollama](https://ollama.com/). Start Ollama before launching the application.

### 2. Open the project folder

Open the repository folder in VS Code, then open **Terminal → New Terminal**.

### 3. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use the following command for the current terminal session, subject to your computer's security policies:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

If the repository contains `requirements.txt`, run:

```powershell
pip install -r requirements.txt
```

Otherwise, install the libraries used by the current app:

```powershell
pip install streamlit chromadb ollama pypdf pandas plotly
```

Additional packages may be needed if the code imports libraries not listed above.

### 5. Download the local models

Run these commands in the terminal:

```powershell
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

The models require local disk space and system memory. The first run may take longer while models are downloaded and loaded.

### 6. Launch the application

From the project root, run:

```powershell
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser. Use the app's upload and indexing controls to add the documents you want to query.

To open the analytics dashboard directly, run:

```powershell
streamlit run pages/Analytics_Dashboard.py
```

The report extractor is available as a page in the main multi-page Streamlit application when the `pages` directory is present.

## Evaluation and validation

The repository includes evaluation scripts for checking selected business questions and answer behavior. From the project root, start with:

```powershell
python evaluate_answers.py
```

Review the script output and expected-answer cases before interpreting the result as a model-wide performance score. A small, rule-based evaluation set is useful for regression checks, but it does not measure all forms of accuracy, retrieval quality, or robustness.

When adding test cases, include questions whose answers are present in the source, questions whose answers are absent, and cases where source figures conflict. For numerical business answers, compare the returned result with the underlying source values rather than judging the wording alone.

## Design choices

**Local-first inference.** Using Ollama makes experimentation possible without a hosted LLM API key and keeps the model runtime under the user's control. Local execution still depends on how the machine and its software are configured; it does not automatically provide user access controls or a secure multi-user deployment.

**Retrieval before generation.** RAG supplies relevant document passages to the language model instead of asking it to answer solely from its pretrained knowledge. This can improve grounding, but it does not eliminate hallucinations or retrieval failures.

**Citations and review.** Source references help a user validate an answer. For financial reporting and other consequential uses, check the source itself before relying on the result.

**Rules for selected metrics.** Clear, structured numerical comparisons are handled with deterministic logic where supported. When input data is ambiguous or conflicting, the system should not invent certainty.

**Human-in-the-loop extraction.** Report extraction can save manual effort, but extracted fields should be reviewed and reconciled against the source report before they are treated as final.

## Known limitations

- Results depend on document parsing, chunking, retrieval, prompt design, and local model quality.
- Scanned PDFs may need OCR, which is not guaranteed by the current workflow.
- Deterministic business logic covers specific known patterns and should be extended and tested before use with new report formats.
- The sample evaluation suite is small and is not a formal benchmark.
- The application is a local prototype; authentication, role-based access, production monitoring, and hardened deployment are outside its current scope.
- Processing sensitive or confidential reports requires appropriate data-handling controls. Do not upload documents unless you are authorized to use them.

## Data and privacy

The included sales files and report examples are intended as demonstration data. Replace them only with data you are authorized to process. The vector database, virtual environment, secrets, private uploaded reports, and backup CSV files should stay out of version control. Check `.gitignore` and `git status` before committing changes.

## Future improvements

- Expand the evaluation dataset with more business questions and difficult edge cases.
- Improve retrieval evaluation, including chunking strategies and measures of source relevance.
- Add stronger structured-data validation and reconciliation for more report layouts.
- Add unit tests for the business rules and extraction checks.
- Improve error handling, document status visibility, and user feedback during indexing.
- Add deployment safeguards before exposing the application to other users.

## Author's note

I built Nexus AI to bring together several parts of an analyst's workflow: finding evidence in reports, answering questions in natural language, validating numerical results, and turning narrative information into data that can be reviewed. The main lesson from the project is that a useful business assistant needs more than a language model. It needs traceable sources, clear rules for numerical questions, validation, and a process for human review.

This repository documents an ongoing project. Features and limitations should be assessed from the code in the repository rather than assumed from the description alone.
