"""Tests for cover-letter localization and salutation de-duplication.

Covers three rendering/localization fixes revealed in the French PDFs:

  1. Duplicate salutation: the greeting/sign-off the LLM may emit inside the
     sentence array is stripped so it is not printed twice.
  2. Hardcoded English strings in French letters: date, recipient, and sign-off
     are localized when language == 'fr'.
  3. Literal '\\$' escaping artifact: '\\$0' in a source PDF is normalized to
     '$0' during extraction so it renders as a clean '$'.
"""

import datetime

import pytest

import core
import latex_builder
from core import (
    _normalize_pdf_artifacts,
    _normalize_cover_letter_body,
    _strip_leading_greeting,
    _strip_trailing_signoff,
    generate_cover_letter,
)
from latex_builder import (
    _localized_date,
    _strip_leading_greeting as lb_strip_greeting,
    render_letter_latex,
)
from models import (
    CoverLetter,
    CoverLetterDraft,
    CoverLetterSentence,
    MasterCV,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _master_cv() -> MasterCV:
    return MasterCV(
        personal_info={
            "name": "Alex Taylor",
            "email": "alex@example.com",
            "linkedin": "linkedin.com/in/alex",
            "github": "github.com/alex",
            "phone": None,
            "location": None,
        },
        facts={"years_experience_total": 3},
        skills=[{"id": "sk_python", "en": "Python", "fr": "Python"}],
    )


def _draft(
    *,
    language: str = "en",
    salutation: str = "Dear Hiring Team,",
    signoff: str = "Sincerely,",
    sentences=None,
) -> CoverLetterDraft:
    if sentences is None:
        sentences = [
            {
                "paragraph": 1,
                "text": "I am excited to apply for this role.",
                "based_on": [],
            }
        ]
    return CoverLetterDraft(
        language=language,
        company="Acme Corp",
        target_role="Backend Engineer",
        salutation=salutation,
        sentences=sentences,
        signoff=signoff,
    )


def _as_full_letter(draft: CoverLetterDraft) -> CoverLetter:
    return CoverLetter(**draft.model_dump())


# ---------------------------------------------------------------------------
# Issue 1: duplicate salutation / sign-off de-duplication
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "greeting",
    [
        "Madame, Monsieur,",
        "Monsieur,",
        "Madame,",
        "Dear Hiring Team,",
        "Dear Ms. Smith,",
        "To whom it may concern,",
    ],
)
def test_strip_leading_greeting_core(greeting):
    raw = f"{greeting} I am writing to apply for the position."
    cleaned = _strip_leading_greeting(raw)
    assert not cleaned.lower().startswith(greeting.lower())
    assert cleaned.startswith("I am writing")


def test_strip_leading_greeting_does_not_touch_body():
    body = "Throughout my career I have delivered reliable systems."
    assert _strip_leading_greeting(body) == body


@pytest.mark.parametrize(
    "signoff",
    ["Cordialement,", "Sincerely,", "Best regards,", "Respectueusement,"],
)
def test_strip_trailing_signoff_core(signoff):
    raw = f"{signoff} Alex Taylor"
    cleaned = _strip_trailing_signoff(raw)
    assert signoff.rstrip(",").lower() not in cleaned.lower() or "alex" in cleaned.lower()


def test_normalize_body_removes_embedded_greeting_and_signoff():
    draft = _draft(
        salutation="Madame, Monsieur,",
        sentences=[
            {"paragraph": 1, "text": "Madame, Monsieur, je vous écris.", "based_on": []},
            {"paragraph": 2, "text": "Fort de mon expérience, je propose.", "based_on": ["sk_python"]},
            {"paragraph": 3, "text": "Cordialement,", "based_on": []},
        ],
    )
    normalized = _normalize_cover_letter_body(draft)
    texts = [s.text for s in normalized.sentences]
    # Greeting stripped from first sentence.
    assert not any(t.lower().startswith("madame") for t in texts)
    # Sign-off-only sentence dropped entirely.
    assert not any(t.lower().startswith("cordialement") for t in texts)
    # The real body sentence survives intact.
    assert any("Fort de mon expérience" in t for t in texts)


def test_normalize_body_is_idempotent():
    draft = _draft(
        salutation="Dear Hiring Team,",
        sentences=[
            {"paragraph": 1, "text": "Dear Hiring Team, I apply.", "based_on": []}
        ],
    )
    once = _normalize_cover_letter_body(draft)
    twice = _normalize_cover_letter_body(once)
    assert [s.text for s in once.sentences] == [s.text for s in twice.sentences]


def test_render_letter_does_not_duplicate_salutation():
    cv = _master_cv()
    letter = _as_full_letter(
        _draft(
            language="fr",
            salutation="Madame, Monsieur,",
            signoff="Cordialement,",
            sentences=[
                {
                    "paragraph": 1,
                    "text": "Madame, Monsieur, je vous écris pour le poste.",
                    "based_on": [],
                },
                {
                    "paragraph": 2,
                    "text": "Fort de mon expérience, je propose une approche.",
                    "based_on": ["sk_python"],
                },
            ],
        )
    )
    # Emulate the pipeline normalization then render.
    letter = _as_full_letter(_normalize_cover_letter_body(letter))
    tex = render_letter_latex(letter, cv)
    # Body text must not repeat the greeting.
    assert tex.count("je vous") == 1
    assert "Madame, Monsieur, je vous" not in tex
    # The salutation itself appears exactly once (rendered by the template).
    assert tex.count("Madame, Monsieur,") == 1


def test_generate_cover_letter_pipeline_strips_greeting(monkeypatch):
    """End-to-end: even if the LLM returns a greeting in the sentences, it's removed."""
    raw_payload = {
        "language": "fr",
        "company": "Acme Corp",
        "target_role": "Ingénieur Backend",
        "salutation": "Madame, Monsieur,",
        "sentences": [
            {"paragraph": 1, "text": "Madame, Monsieur, je vous écris.", "based_on": []},
            {
                "paragraph": 2,
                "text": "Je maîtrise Python pour la production.",
                "based_on": ["sk_python"],
            },
        ],
        "signoff": "Cordialement, Alex Taylor",
    }
    monkeypatch.setattr(core, "_call_llm_json", lambda *a, **k: raw_payload)

    letter = generate_cover_letter(
        "Backend role", _master_cv(), api_key="fake", language="fr"
    )
    body = " ".join(s.text for s in letter.sentences)
    assert not body.lower().startswith("madame")
    assert "Madame, Monsieur, je vous" not in body


# ---------------------------------------------------------------------------
# Issue 2: French localization of date, recipient, sign-off
# ---------------------------------------------------------------------------


def test_localized_date_french():
    d = datetime.date(2026, 11, 13)
    assert _localized_date("fr", d) == "13 novembre 2026"


def test_localized_date_english():
    d = datetime.date(2026, 11, 13)
    assert _localized_date("en", d) == "November 13, 2026"


def test_localized_date_all_french_months():
    for month in range(1, 13):
        d = datetime.date(2026, month, 1)
        out = _localized_date("fr", d)
        # No English month names should leak through.
        for en_month in (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ):
            assert en_month not in out


def test_french_letter_uses_localized_recipient_and_signoff(monkeypatch):
    monkeypatch.setattr(
        latex_builder, "_localized_date", lambda lang, today=None: _localized_date("fr")
    )
    cv = _master_cv()
    letter = _as_full_letter(
        _draft(
            language="fr",
            salutation="Madame, Monsieur,",
            signoff="Cordialement",
            sentences=[
                {"paragraph": 1, "text": "Je postule pour ce poste.", "based_on": []}
            ],
        )
    )
    tex = render_letter_latex(letter, cv)
    assert "Équipe de recrutement" in tex
    assert "Hiring Team" not in tex
    # French sign-off present, English default absent.
    assert "Cordialement" in tex
    assert "Sincerely" not in tex


def test_english_letter_keeps_english_strings(monkeypatch):
    cv = _master_cv()
    letter = _as_full_letter(
        _draft(
            language="en",
            salutation="Dear Hiring Team,",
            signoff="Sincerely,",
            sentences=[{"paragraph": 1, "text": "I apply for this role.", "based_on": []}],
        )
    )
    tex = render_letter_latex(letter, cv)
    assert "Hiring Team" in tex
    assert "Sincerely" in tex
    assert "Équipe de recrutement" not in tex


def test_french_letter_default_signoff_when_model_signoff_is_name(monkeypatch):
    """A model-provided sign-off that's just the candidate name falls back to localized default."""
    cv = _master_cv()
    letter = _as_full_letter(
        _draft(
            language="fr",
            salutation="Madame, Monsieur,",
            signoff="Alex Taylor",
            sentences=[{"paragraph": 1, "text": "Je postule.", "based_on": []}],
        )
    )
    tex = render_letter_latex(letter, cv)
    # Should fall back to the localized French sign-off, not English "Sincerely".
    assert "Cordialement" in tex
    assert "Sincerely" not in tex


# ---------------------------------------------------------------------------
# Issue 3: backslash-dollar artifact normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("\\$0", "$0"),
        ("pays \\$0 per request", "pays $0 per request"),
        ("cost is R\\$ 5", "cost is R$ 5"),
    ],
)
def test_normalize_pdf_artifacts(raw, expected):
    assert _normalize_pdf_artifacts(raw) == expected


def test_backslash_dollar_renders_clean_dollar():
    """After normalization + LaTeX escaping, a single escaped '$' results (clean)."""
    normalized = _normalize_pdf_artifacts("Priced at \\$0")
    escaped = latex_builder.escape_latex(normalized)
    # Exactly one escaped dollar, and NO leftover raw backslash before it.
    assert "\\$0" in escaped.replace("\\\\", "")  # single '\$' present
    assert normalized == "Priced at $0"


def test_normalize_pdf_artifacts_empty_and_none():
    assert _normalize_pdf_artifacts("") == ""
    assert _normalize_pdf_artifacts(None) is None


# ---------------------------------------------------------------------------
# Cross-check: core and latex_builder greeting strippers agree
# ---------------------------------------------------------------------------


def test_greeting_strippers_agree():
    samples = [
        "Madame, Monsieur, body text",
        "Dear Hiring Team, body text",
        "Monsieur, body text",
    ]
    for s in samples:
        assert _strip_leading_greeting(s) == lb_strip_greeting(s)
