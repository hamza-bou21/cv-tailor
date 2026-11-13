"""Tests for the PDF -> master_cv.yaml bootstrap pipeline.

These use a synthetic in-memory PDF (real PyMuPDF geometry) plus a mocked LLM
response, so they run offline and deterministically. They guard the six
extraction/integrity fixes:

  1. no fabricated / padded / duplicated bullets (exact counts preserved)
  2. reading-order + de-hyphenation of PDF text
  3. non-truncated projects & education sections
  4. embedded hyperlinks mapped to the right fields
  5. deterministic post-extraction grounding warnings
  6. single shared extraction entrypoint for CLI and UI
"""

import pymupdf
import pytest

import core
from core import (
    build_master_cv_from_pdf,
    extract_text_from_pdf,
    extract_pdf_content,
    ground_extraction,
    _dehyphenate,
)
from models import MasterCV


def _make_resume_pdf() -> bytes:
    """Builds a one-page synthetic CV reproducing the reported PDF artifacts."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)

    # --- Experience with a right-aligned date on the same baseline ---
    page.insert_text((50, 80), "WORK EXPERIENCE", fontsize=12)
    page.insert_text((50, 110), "Backend Engineer - TechCorp", fontsize=10)
    page.insert_text((430, 110), "01/2021 - 12/2023", fontsize=10)
    page.insert_text((50, 130), "- Built pipeline handling 8000 rps", fontsize=10)
    page.insert_text((50, 150), "- Used Chro-", fontsize=10)
    page.insert_text((50, 165), "maDB vector store", fontsize=10)

    # --- Education with a right-aligned date (was sinking to the bottom) ---
    page.insert_text((50, 200), "EDUCATION", fontsize=12)
    page.insert_text((50, 230), "MSc Computer Science, Tech Institute", fontsize=10)
    page.insert_text((450, 230), "09/2023 - 06/2025", fontsize=10)

    # --- Adjacent date-like tokens on the same baseline (merge trap) ---
    page.insert_text((50, 260), "11/2025", fontsize=10)
    page.insert_text((105, 260), "11/2026", fontsize=10)

    # --- Projects with a 'Live demo' hyperlink anchor ---
    page.insert_text((50, 300), "PROJECTS", fontsize=12)
    page.insert_text((50, 330), "E-Commerce Analytics (Olist)", fontsize=10)
    page.insert_text((50, 345), "Live demo", fontsize=10)
    page.insert_link(
        {
            "kind": pymupdf.LINK_URI,
            "from": pymupdf.Rect(50, 335, 120, 350),
            "uri": "https://github.com/user/olist-demo",
        }
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# ---------------------------------------------------------------------------
# Fix 2: reading order + de-hyphenation
# ---------------------------------------------------------------------------


def test_right_aligned_dates_stay_with_their_entries():
    text = extract_text_from_pdf(_make_resume_pdf())
    # Education line and its right-aligned date must both be present and adjacent
    # in the stream (education not dropped/sunk to the bottom).
    edu_idx = text.index("MSc Computer Science")
    date_idx = text.index("09/2023 - 06/2025")
    assert date_idx > edu_idx
    # The date should be on the very next line after the education line.
    assert (
        text.splitlines()[
            text.splitlines().index("MSc Computer Science, Tech Institute") + 1
        ].strip()
        == "09/2023 - 06/2025"
    )


def test_adjacent_dates_are_not_merged():
    text = extract_text_from_pdf(_make_resume_pdf())
    assert "11/202511/2026" not in text
    assert ("11/2025 11/2026" in text) or ("11/2025\n11/2026" in text)


def test_soft_line_break_hyphen_is_melded():
    text = extract_text_from_pdf(_make_resume_pdf())
    assert "ChromaDB" in text
    assert "Chro-\nmaDB" not in text


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Built with Chro-\nmaDB and Redis.", "Built with ChromaDB and Redis."),
        ("Text-\nSQL pipelines", "Text-SQL pipelines"),
        ("Scikit-\nlearn models", "Scikit-learn models"),
        ("cross-\nfunctional teams", "cross-functional teams"),
    ],
)
def test_dehyphenate_preserves_legit_compounds(raw, expected):
    assert _dehyphenate(raw) == expected


# ---------------------------------------------------------------------------
# Fix 4: hyperlink anchoring
# ---------------------------------------------------------------------------


def test_hyperlink_anchored_to_visible_text():
    _, links = extract_pdf_content(_make_resume_pdf())
    urls = {l.url: l.anchor_text for l in links}
    assert "https://github.com/user/olist-demo" in urls
    assert "Live demo" in urls["https://github.com/user/olist-demo"]


# ---------------------------------------------------------------------------
# Fixes 1, 3, 5, 6: mocked LLM -> exact counts, no fabrication, grounding,
# shared entrypoint.
# ---------------------------------------------------------------------------


def _mock_llm_payload():
    """A faithful extraction of the synthetic PDF (exact bullet counts)."""
    return {
        "personal_info": {
            "name": "Alex Taylor",
            "email": "alex@example.com",
            "linkedin": "linkedin.com/in/alex",
            "github": "github.com/alex",
        },
        "facts": {"years_experience_total": 3},
        "skills": [{"id": "sk_pymupdf", "en": "PyMuPDF", "fr": "PyMuPDF"}],
        "experience": [
            {
                "id": "exp_techcorp",
                "company": "TechCorp",
                "location": None,
                "role": {"en": "Backend Engineer", "fr": "Ingénieur Backend"},
                "start_date": "01/2021",
                "end_date": "12/2023",
                # EXACTLY 2 bullets, as in the source PDF (no padding to 3).
                "bullets": [
                    {
                        "id": "techcorp_b1",
                        "en": "Built pipeline handling 8000 rps",
                        "fr": "Pipeline gérant 8000 rps",
                    },
                    {
                        "id": "techcorp_b2",
                        "en": "Used ChromaDB vector store",
                        "fr": "Stockage vectoriel ChromaDB",
                    },
                ],
            }
        ],
        "projects": [
            {
                "id": "proj_olist",
                "name": "E-Commerce Analytics (Olist)",
                "url": "https://github.com/user/olist-demo",
                "description": {
                    "en": "E-commerce analytics.",
                    "fr": "Analyse e-commerce.",
                },
                "bullets": [
                    {
                        "id": "olist_b1",
                        "en": "Analyzed order data with 8000 rows",
                        "fr": "Analyse de 8000 lignes",
                    }
                ],
            }
        ],
        "education": [
            {
                "id": "edu_msc",
                "institution": "Tech Institute",
                "degree": {
                    "en": "MSc Computer Science",
                    "fr": "Master Informatique",
                },
                "graduation_year": "2025",
            }
        ],
    }


def test_extraction_preserves_counts_and_projects(monkeypatch):
    monkeypatch.setattr(core, "_call_llm_json", lambda *a, **k: _mock_llm_payload())
    cv, warnings = build_master_cv_from_pdf(_make_resume_pdf(), api_key="fake")

    # Fix 1: exact bullet counts preserved.
    assert len(cv.experience) == 1
    assert len(cv.experience[0].bullets) == 2

    # Fix 3: project + education present (not truncated).
    assert len(cv.projects) == 1
    assert cv.projects[0].name == "E-Commerce Analytics (Olist)"
    assert len(cv.education) == 1

    # Fix 4: project url mapped.
    assert cv.projects[0].url == "https://github.com/user/olist-demo"

    # No grounding warnings for a faithful extraction.
    assert warnings == []


def test_grounding_flags_fabricated_number(monkeypatch):
    payload = _mock_llm_payload()
    # Inject a fabricated metric not present in the synthetic PDF.
    payload["experience"][0]["bullets"][0]["en"] = "Boosted throughput by 42000 rps"
    monkeypatch.setattr(core, "_call_llm_json", lambda *a, **k: payload)

    cv, warnings = build_master_cv_from_pdf(_make_resume_pdf(), api_key="fake")
    assert any("42000" in w for w in warnings), warnings


def test_grounding_flags_empty_education_when_degrees_present(monkeypatch):
    payload = _mock_llm_payload()
    payload["education"] = []  # drop education despite 'MSc'/'Institute' in PDF
    monkeypatch.setattr(core, "_call_llm_json", lambda *a, **k: payload)

    cv, warnings = build_master_cv_from_pdf(_make_resume_pdf(), api_key="fake")
    assert cv.education == []
    assert any("Education" in w for w in warnings), warnings


def test_ground_extraction_direct():
    cv = MasterCV(**_mock_llm_payload())
    # Source text lacks the project's '8000' metric -> should be flagged.
    warn = ground_extraction(cv, "Backend Engineer TechCorp, MSc Computer Science")
    assert any("8000" in w for w in warn)
