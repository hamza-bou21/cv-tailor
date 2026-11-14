# CV Tailor

[![Tests](https://github.com/hamza-bou21/cv-tailor/actions/workflows/tests.yml/badge.svg)](https://github.com/hamza-bou21/cv-tailor/actions/workflows/tests.yml)

> An open-source CV tailoring tool that can't fabricate experience: every claim in the output traces back to your real CV, with deterministic checks for fabricated numbers.

---

## Overview

CV Tailor ingests a master CV and a target job posting, produces an objective fit assessment, and generates a tailored one-page CV and cover letter (EN/FR) compiled from LaTeX. Every generated claim traces back to unique identifiers within your master CV, and unsupported claims and fabricated metrics are flagged by deterministic checks.

Master data, databases and generated documents stay local to your machine. Note that the CV and job text you analyze are sent to the model provider (Groq) for processing.

---

## Status: full pipeline shipped

- [x] **Ingestion** — extract raw text, reading-ordered layout, and embedded hyperlinks from CV PDFs using `pymupdf` into a validated, bilingual `master_cv.yaml` with schema-enforced Pydantic models. Grouped skills are split into atomic entries with bilingual categories, spoken-language proficiency is preserved, and a grounded professional summary is captured (never invented).
- [x] **Fit engine** — analyze raw job postings against `master_cv.yaml` via Groq (`openai/gpt-oss-120b`). Outputs a structured fit score (0 to 100), a verdict (`apply` / `stretch` / `skip`), confirmed strengths, and a gap analysis, with any skill not in the master CV discarded.
- [x] **Ranking-based selection** — the LLM *ranks* every skill / experience bullet / project bullet by relevance (most → least) rather than filtering, so the layout engine — not the model — decides what gets trimmed. Hallucinated IDs are dropped and any omitted valid items are re-appended deterministically.
- [x] **One-page compilation** — LaTeX templates styled after a clean single-column layout, grouped skills, a grounded summary block, and a two-stage one-page overflow loop: typography/spacing compaction first (down to a 9.5pt readability floor), then deterministic lowest-rank pruning.
- [x] **Cover letter** — cite-then-validate engine with automatic re-prompting on failed checks, a compact single-language context payload, and localised salutation / date / sign-off.
- [x] **Mindful of provider limits** — per-call `max_tokens` budgets and `reasoning_effort` capping for `gpt-oss` / `qwen` reasoning models keep each request inside Groq's 8,000 TPM ceiling without truncating JSON.

---

## Architecture and core rules

1. **Rank-then-trim, cite-then-validate** — the LLM does not write new bullet points. It selects existing bullets and skills by ID only and ranks them by relevance; any hallucinated ID is dropped deterministically.
2. **Deterministic metric and skill checks** — numerals and named skills are validated via regex and set comparison against the `facts` block, experience/project bullets, and skill list — not by trusting model self-reflection.
3. **Layout-budgeted content** — content is preserved through typography compaction and only trimmed (lowest-ranked first) as a last resort at the readability floor, so the strongest material survives.
4. **Local-first and private** — master data (`*.yaml`), the SQLite database (`*.db`), and compiled `*.pdf` / `*.tex` artifacts are ignored by version control.
5. **Provenance tracking** — selection edits do not trigger re-verification. Manual text edits set `manually_edited = true` to preserve audit integrity.

---

## Project structure

```text
cv-tailor/
├── .env.example          # Template for required environment variables
├── .gitignore            # Git ignore rules for privacy and build artifacts
├── LICENSE               # MIT License
├── README.md             # Project documentation
├── REQUIREMENTS.md       # Consolidated product specifications
├── requirements.txt      # Python dependencies
├── demo/
│   ├── master_cv.yaml    # Sample master CV for trying things out
│   └── posting.txt       # Sample job posting
├── tests/
│   ├── test_selection.py                 # Deterministic ID-scrubbing tests
│   ├── test_verification.py              # Numeral / skill / citation verification tests
│   ├── test_pdf_extraction.py            # PDF -> master_cv extraction & grounding tests
│   ├── test_layout_and_summaries.py      # Selection ranking, layout, grouped skills & summaries
│   └── test_cover_letter_localization.py # Localization, TPM budget & reasoning-model tests
├── models.py             # Pydantic data schemas and verification contracts
├── core.py               # Fit analysis, selection, verification, PDF parsing
├── latex_builder.py      # LaTeX CV & cover-letter rendering + PDF compilation
├── db.py                 # SQLite audit log and history helpers
├── init_cv.py            # Bootstrap script to parse a PDF into master_cv.yaml
├── analyze.py            # Command-line pipeline (analysis -> CV -> cover letter)
└── app.py                # Streamlit web interface
```

---

## Getting started

### Prerequisites

- Python 3.10+
- A [Groq Cloud](https://console.groq.com/) API key
- A LaTeX distribution providing `pdflatex` (MiKTeX or TeX Live). Required for PDF compilation.

  > On Windows, `latex_builder.py` auto-detects `pdflatex` from PATH or the default MiKTeX install locations.

### Model & provider notes

- The default model is `openai/gpt-oss-120b`. Reasoning models like `gpt-oss` and `qwen` spend part of `max_tokens` on internal reasoning before emitting JSON, so the client caps their reasoning (`reasoning_effort="low"` / `"none"`) and sizes `max_tokens` per call to stay under Groq's 8,000 TPM single-request ceiling. Plain models such as `llama-3.3-70b-versatile` are called without `reasoning_effort` (they reject the parameter).
- Rate/token-limit errors (`429` / `413`) trigger a 10-second backoff before retrying, giving the rolling TPM window time to clear.

### Installation

1. Clone the repository:

```bash
git clone https://github.com/hamza-bou21/cv-tailor.git
cd cv-tailor
```

2. Set up a virtual environment:

```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure secrets:

```bash
cp .env.example .env
```

Open `.env` and add your Groq API key:

```text
GROQ_API_KEY=gsk_your_actual_key_here
```

---

## Usage

There are two ways to use CV Tailor: the **Streamlit web app** (recommended) or the **command-line pipeline**.

### Option A — Streamlit web app

```bash
streamlit run app.py
```

The app has three tabs:

- **Job Tailoring** — paste a posting, run the fit analysis, and download the tailored CV and cover letter.
- **Sources & Profile Manager** — import a CV from PDF, and inspect/edit/delete saved `master_cv.yaml` profiles.
- **Provenance & Audit History** — browse every generated document with its SHA-256 hash, and delete individual records or purge all history.

### Option B — Command-line pipeline

```bash
# Analyze a posting against the demo CV and compile a CV with a cover letter
python analyze.py --job demo/posting.txt --letter

# Use a specific profile, French output, and a named company
python analyze.py --cv profiles/jane_doe.yaml --job posting.txt --lang fr --company "Acme Inc"

# Generate even when the fit verdict is SKIP
python analyze.py --force
```

Available flags:

| Flag | Description |
|------|-------------|
| `--cv` | Path to master CV YAML (falls back to `demo/master_cv.yaml`) |
| `--job` | Path to job posting text file |
| `--lang` | Target language: `en` or `fr` (default `en`) |
| `--letter` | Also generate a verified cover letter |
| `--company` | Company name recorded in the audit log |
| `--force` | Generate even if the fit verdict is `SKIP` |

### Bootstrap your master CV from a PDF

Extract text, hyperlinks and experience metrics from an existing PDF into a structured `master_cv.yaml`:

```bash
python init_cv.py "path/to/your_cv.pdf"
```

---

## Testing

Run the deterministic test suite (selection, verification, PDF extraction, layout/summaries, and cover-letter localization — all offline and API-key free):

```bash
pytest
```

---

## License

Distributed under the [MIT License](LICENSE).