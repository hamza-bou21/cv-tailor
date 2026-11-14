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
    _build_cover_letter_catalog,
    generate_cover_letter,
)
from latex_builder import (
    _localized_date,
    _strip_leading_greeting as lb_strip_greeting,
    render_letter_latex,
)
from models import (
    FitAnalysis,
    CoverLetter,
    CoverLetterDraft,
    CoverLetterSentence,
    CVSelection,
    MasterCV,
    SelectedExperience,
    SelectedProject,
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
    assert (
        signoff.rstrip(",").lower() not in cleaned.lower() or "alex" in cleaned.lower()
    )


def test_normalize_body_removes_embedded_greeting_and_signoff():
    draft = _draft(
        salutation="Madame, Monsieur,",
        sentences=[
            {
                "paragraph": 1,
                "text": "Madame, Monsieur, je vous écris.",
                "based_on": [],
            },
            {
                "paragraph": 2,
                "text": "Fort de mon expérience, je propose.",
                "based_on": ["sk_python"],
            },
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
            {
                "paragraph": 1,
                "text": "Madame, Monsieur, je vous écris.",
                "based_on": [],
            },
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
            sentences=[
                {"paragraph": 1, "text": "I apply for this role.", "based_on": []}
            ],
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


# ---------------------------------------------------------------------------
# Issue 4: 8,000 TPM budget - compact, single-language, focused cover-letter
# context and right-sized max_tokens.
# ---------------------------------------------------------------------------


def _bilingual_cv(n_skills: int = 24, n_experiences: int = 3, bullets_each: int = 3):
    """Builds a MasterCV with distinct EN/FR strings to detect language leakage."""
    skills = [
        {
            "id": f"sk_{i}",
            "en": f"ENSkill{i}",
            "fr": f"FRSkill{i}",
            "category": {"en": "Technologies", "fr": "TechnologiesFR"},
        }
        for i in range(n_skills)
    ]
    experience = []
    for e in range(n_experiences):
        experience.append(
            {
                "id": f"exp_{e}",
                "company": f"Company{e}",
                "location": None,
                "role": {"en": f"ENRole{e}", "fr": f"FRRole{e}"},
                "start_date": "2020",
                "end_date": "2023",
                "bullets": [
                    {
                        "id": f"exp_{e}_b{b}",
                        "en": f"ENBullet{e}_{b}",
                        "fr": f"FRBullet{e}_{b}",
                    }
                    for b in range(bullets_each)
                ],
            }
        )
    projects = [
        {
            "id": "proj_0",
            "name": "ProjName",
            "url": None,
            "description": {"en": "ENDesc", "fr": "FRDesc"},
            "bullets": [
                {"id": "proj_0_b0", "en": "ENProjBullet0", "fr": "FRProjBullet0"},
                {"id": "proj_0_b1", "en": "ENProjBullet1", "fr": "FRProjBullet1"},
                {"id": "proj_0_b2", "en": "ENProjBullet2", "fr": "FRProjBullet2"},
            ],
        }
    ]
    summaries = [
        {"id": "sum_default", "en": "ENSummary text.", "fr": "FRSummary texte."}
    ]
    return MasterCV(
        personal_info={
            "name": "Alex Taylor",
            "email": "alex@example.com",
            "linkedin": "linkedin.com/in/alex",
            "github": "github.com/alex",
        },
        facts={"years_experience_total": 5},
        skills=skills,
        experience=experience,
        projects=projects,
        summaries=summaries,
    )


class _SpyLLM:
    """Records the messages/kwargs of the last _call_llm_json invocation."""

    def __init__(self, payload):
        self.payload = payload
        self.messages = None
        self.kwargs = None
        self.calls = 0

    def __call__(self, messages, schema, api_key, model, **kwargs):
        self.calls += 1
        self.messages = messages
        self.kwargs = kwargs
        return self.payload


def _valid_letter_payload(language: str) -> dict:
    return {
        "language": language,
        "company": "Acme Corp",
        "target_role": "Backend Engineer",
        "salutation": "Dear Hiring Team," if language == "en" else "Madame, Monsieur,",
        "sentences": [
            {
                "paragraph": 1,
                "text": "I am excited to apply.",
                "based_on": [],
            }
        ],
        "signoff": "Sincerely," if language == "en" else "Cordialement,",
    }


def _full_selection(cv: MasterCV) -> CVSelection:
    return CVSelection(
        language="en",
        target_role="Backend Engineer",
        skill_ids=[s.id for s in cv.skills],
        experiences=[
            SelectedExperience(experience_id=e.id, bullet_ids=[b.id for b in e.bullets])
            for e in cv.experience
        ],
        projects=[
            SelectedProject(project_id=p.id, bullet_ids=[b.id for b in p.bullets])
            for p in cv.projects
        ],
        summary_id=cv.summaries[0].id if cv.summaries else None,
    )


def _payload_text(messages) -> str:
    return " ".join(m["content"] for m in messages)


@pytest.mark.parametrize("language", ["en", "fr"])
def test_cover_letter_sends_only_target_language(monkeypatch, language):
    """The prompt must contain the active language's strings and none of the other."""
    cv = _bilingual_cv()
    selection = _full_selection(cv)
    spy = _SpyLLM(_valid_letter_payload(language))
    monkeypatch.setattr(core, "_call_llm_json", spy)

    generate_cover_letter(
        "Backend role",
        cv,
        api_key="fake",
        language=language,
        selection=selection,
    )

    payload = _payload_text(spy.messages)
    if language == "en":
        # Active-language strings present; inactive FR strings absent.
        assert "ENSkill0" in payload
        assert "ENBullet0_0" in payload
        assert "ENRole0" in payload
        assert "FRSkill0" not in payload
        assert "FRBullet0_0" not in payload
        assert "FRRole0" not in payload
        assert "FRSummary texte." not in payload
    else:
        assert "FRSkill0" in payload
        assert "FRBullet0_0" in payload
        assert "FRRole0" in payload
        assert "ENSkill0" not in payload
        assert "ENBullet0_0" not in payload
        assert "ENRole0" not in payload
        assert "ENSummary text." not in payload


def test_cover_letter_caps_skills_at_12_but_keeps_top2_experience_bullets(monkeypatch):
    """24+ skills are capped to 12, while top-2 experiences keep ALL their bullets."""
    cv = _bilingual_cv(n_skills=24, n_experiences=3, bullets_each=3)
    selection = _full_selection(cv)
    spy = _SpyLLM(_valid_letter_payload("en"))
    monkeypatch.setattr(core, "_call_llm_json", spy)

    generate_cover_letter(
        "Backend role",
        cv,
        api_key="fake",
        language="en",
        selection=selection,
    )

    payload = _payload_text(spy.messages)
    # Exactly the first 12 ranked skills are present; 13th+ are dropped.
    for i in range(12):
        assert f"ENSkill{i}" in payload, f"skill {i} should be present"
    for i in range(12, 24):
        assert f"ENSkill{i}" not in payload, f"skill {i} should be capped out"

    # Top-2 experiences keep ALL of their bullets.
    assert "ENBullet0_0" in payload and "ENBullet0_2" in payload
    assert "ENBullet1_0" in payload and "ENBullet1_2" in payload
    # The 3rd experience is capped at its top 2 bullets (b2 dropped).
    assert "ENBullet2_0" in payload and "ENBullet2_1" in payload
    assert "ENBullet2_2" not in payload


def test_cover_letter_catalog_skill_cap_direct():
    """_build_cover_letter_catalog caps skills to 12 and resolves one language."""
    cv = _bilingual_cv(n_skills=30)
    selection = _full_selection(cv)
    catalog = _build_cover_letter_catalog(cv, "en", selection=selection)
    assert len(catalog["skills"]) == 12
    # Single-language: only 'name' keys with EN strings, no FR leakage.
    assert all("name" in s and "fr" not in s for s in catalog["skills"])
    assert catalog["skills"][0]["name"] == "ENSkill0"


def test_cover_letter_passes_max_tokens_within_budget(monkeypatch):
    """_call_llm_json must be invoked with a max_tokens that fits the TPM budget.

    Reasoning models (gpt-oss) spend part of max_tokens on internal reasoning, so
    the cover letter uses 2600 to leave headroom for JSON without truncation.
    """
    cv = _bilingual_cv()
    spy = _SpyLLM(_valid_letter_payload("en"))
    monkeypatch.setattr(core, "_call_llm_json", spy)

    generate_cover_letter("Backend role", cv, api_key="fake", language="en")

    assert spy.kwargs is not None
    assert "max_tokens" in spy.kwargs
    assert spy.kwargs["max_tokens"] <= 3000
    assert spy.kwargs["max_tokens"] == 2600


def test_cover_letter_catalog_uses_compact_json(monkeypatch):
    """The serialized catalog must be compact JSON (no indentation whitespace)."""
    cv = _bilingual_cv()
    spy = _SpyLLM(_valid_letter_payload("en"))
    monkeypatch.setattr(core, "_call_llm_json", spy)

    generate_cover_letter("Backend role", cv, api_key="fake", language="en")

    user_msg = next(m["content"] for m in spy.messages if m["role"] == "user")
    # Compact separators mean no '": ' or '", "' spacing produced by indent.
    assert '": ' not in user_msg
    assert '", "' not in user_msg


def test_cover_letter_includes_fit_result_signals(monkeypatch):
    """fit_result strengths / covered_keywords are folded into the context."""
    cv = _bilingual_cv()
    spy = _SpyLLM(_valid_letter_payload("en"))
    monkeypatch.setattr(core, "_call_llm_json", spy)
    fit = FitAnalysis(
        fit_score=88,
        verdict="apply",
        strengths=["distributed systems"],
        gaps=[],
        covered_keywords=["python"],
        missing_keywords=[],
    )

    generate_cover_letter(
        "Backend role", cv, api_key="fake", language="en", fit_result=fit
    )

    payload = _payload_text(spy.messages)
    assert "distributed systems" in payload
    assert "python" in payload


# ---------------------------------------------------------------------------
# Issue 5: reasoning models (gpt-oss / qwen) must receive a capped
# reasoning_effort so internal reasoning tokens don't exhaust max_tokens before
# the JSON document is emitted ("max completion tokens reached before generating
# a valid document"), while plain models must NOT receive the parameter.
# ---------------------------------------------------------------------------


class _FakeGroqClient:
    """Stand-in for groq.Groq that records the kwargs passed to create()."""

    last_create_kwargs = None

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        _FakeGroqClient.last_create_kwargs = kwargs
        # Minimal response object shaped like Groq's: choices[0].message.content
        class _Choice:
            class _Msg:
                content = '{"ok": true}'

            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


@pytest.fixture
def fake_groq(monkeypatch):
    _FakeGroqClient.last_create_kwargs = None
    monkeypatch.setattr(core, "Groq", _FakeGroqClient)
    monkeypatch.setattr(core.time, "sleep", lambda *_: None)
    return _FakeGroqClient


def _call_with_model(model):
    return core._call_llm_json(
        messages=[{"role": "user", "content": "hi"}],
        schema={},
        api_key="fake",
        model=model,
    )


def test_call_llm_json_gpt_oss_passes_reasoning_effort_low(fake_groq):
    """gpt-oss models receive reasoning_effort='low' to cap reasoning tokens."""
    _call_with_model("openai/gpt-oss-120b")
    kwargs = fake_groq.last_create_kwargs
    assert kwargs is not None
    assert kwargs.get("reasoning_effort") == "low"
    # max_tokens headroom is preserved for the JSON document.
    assert kwargs.get("max_tokens") is not None


def test_call_llm_json_omits_reasoning_effort_for_plain_model(fake_groq):
    """Non-reasoning llama models must NOT receive reasoning_effort."""
    _call_with_model("llama-3.3-70b-versatile")
    kwargs = fake_groq.last_create_kwargs
    assert kwargs is not None
    assert "reasoning_effort" not in kwargs


def test_reasoning_kwargs_helper():
    """The helper maps model families to the correct reasoning-cap kwargs."""
    assert core._reasoning_kwargs_for_model("openai/gpt-oss-120b") == {
        "reasoning_effort": "low"
    }
    assert core._reasoning_kwargs_for_model("qwen/qwen3-32b") == {
        "reasoning_effort": "none"
    }
    assert core._reasoning_kwargs_for_model("llama-3.3-70b-versatile") == {}
