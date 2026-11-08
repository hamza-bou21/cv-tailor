# CV Tailor

> An open-source CV tailoring tool that can't fabricate experience: every claim in the output traces back to your real CV, with deterministic checks for fabricated numbers.

---

## Overview

CV Tailor ingests a master CV and a target job posting, produces an objective fit assessment, and generates a tailored one-page CV and cover letter (EN/FR) compiled from LaTeX. Every generated claim traces back to unique identifiers within your master CV, and unsupported claims and fabricated metrics are flagged by deterministic checks.

Master data, databases and generated documents stay local to your machine. Note that the CV text you analyze is sent to the model provider (Groq) for processing.

---

## Status: v0 shipped

- [x] Step 0, Init and ingestion: extract raw text and embedded hyperlinks from CV PDFs using `pymupdf` into a validated, bilingual `master_cv.yaml` with schema-enforced Pydantic constraints.
- [x] v0, Fit engine: analyze raw job postings against `master_cv.yaml` via Groq (`openai/gpt-oss-120b`). Outputs a structured fit score (0 to 100), a verdict (`apply` / `stretch` / `skip`), confirmed strengths, and gap analysis, with any skill not in the master CV discarded.
- [ ] v0.5, Selection and compilation: ID-based bullet selection engine, LaTeX templates, and a one-page overflow reduction loop.
- [ ] v1.0, Full pipeline: cover letter cite-then-validate engine, SQLite history, and Streamlit user interface.

---

## Architecture and core rules

1. Cite-then-validate: the LLM does not write new bullet points. It selects existing bullets and skills by ID only.
2. Deterministic metric and skill checks: numerals and named skills are validated via regex and set comparison against the `facts` block and skill list, not by trusting model self-reflection.
3. Local-first and private: master data, database entries and compiled PDFs are ignored by version control.
4. Provenance tracking: selection edits do not trigger re-verification. Manual text edits set `manually_edited = true` to preserve audit integrity.

---

## Project structure

```text
cv-tailor/
├── .env.example          # Template for required environment variables
├── .gitignore            # Git ignore rules for privacy and build artifacts
├── LICENSE               # MIT License
├── REQUIREMENTS.md       # Consolidated v1 product specifications
├── README.md             # Project documentation
├── requirements.txt      # Python dependencies
├── models.py             # Pydantic data schemas and verification contracts
├── init_cv.py            # Bootstrap script to parse a PDF into master_cv.yaml
├── core.py               # Core analysis and fit evaluation logic
└── test_v0.py            # Verification and test runner for v0
```

---

## Getting started

### Prerequisites

- Python 3.10+
- A [Groq Cloud](https://console.groq.com/) API key
- TeX Live with French language support (required for v0.5+ PDF compilation)

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

### 1. Bootstrap your master CV

Extract text, hyperlinks and experience metrics from an existing PDF into a structured `master_cv.yaml`:

```bash
python init_cv.py "path/to/your_cv.pdf"
```

### 2. Run fit analysis (v0)

Evaluate fit against a target job posting:

```bash
python test_v0.py
```

---

## License

Distributed under the [MIT License](LICENSE).