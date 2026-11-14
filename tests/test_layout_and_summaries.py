"""Tests for the layout-overflow loop, categorized skills, ranked selection, and
grounded summary variants (new tailoring features)."""

import pytest

from core import (
    load_master_cv,
    select_cv_content,
    _normalize_summaries,
    _strip_languages_from_summary,
)
from models import (
    MasterCV,
    Skill,
    SummaryVariant,
    CVSelection,
    SelectedExperience,
    SelectedProject,
    BilingualText,
)
import latex_builder


@pytest.fixture
def demo_cv():
    return load_master_cv("demo/master_cv.yaml")


# ---------------------------------------------------------------------------
# Selection: rank, don't filter (keep all items, append any omitted)
# ---------------------------------------------------------------------------


def _fake_llm_factory(payload):
    """Returns a stub for core._call_llm_json returning a fixed payload."""

    def _fake(*args, **kwargs):
        return payload

    return _fake


def test_selection_keeps_all_experience_and_bullets(monkeypatch, demo_cv):
    """LLM that returns only ONE bullet per experience must not cause data loss."""
    payload = {
        "language": "en",
        "target_role": "Backend Engineer",
        "skill_ids": ["sk_python"],  # only one skill selected
        "experiences": [
            {"experience_id": "exp_techcorp", "bullet_ids": ["techcorp_b2"]}
        ],
        "projects": [],
        "education_ids": [],
        "summary_id": None,
    }
    monkeypatch.setattr("core._call_llm_json", _fake_llm_factory(payload))

    sel = select_cv_content("job", demo_cv, api_key="dummy")

    # Experience block present with ranked bullet first, then omitted bullet appended.
    exp = sel.experiences[0]
    assert exp.experience_id == "exp_techcorp"
    assert exp.bullet_ids == ["techcorp_b2", "techcorp_b1"]

    # Projects omitted by the LLM are appended back (nothing lost pre-layout).
    assert [p.project_id for p in sel.projects] == ["proj_cvtailor"]
    assert sel.projects[0].bullet_ids == ["cvtailor_b1"]

    # Skills: selected skill first, then remaining valid skills appended.
    assert sel.skill_ids[0] == "sk_python"
    assert set(sel.skill_ids) == {s.id for s in demo_cv.skills}

    # Education falls back to all entries when none selected.
    assert sel.education_ids == [e.id for e in demo_cv.education]


def test_selection_drops_hallucinated_ids(monkeypatch, demo_cv):
    payload = {
        "language": "en",
        "target_role": "Backend Engineer",
        "skill_ids": ["sk_python", "fake_skill"],
        "experiences": [
            {
                "experience_id": "exp_techcorp",
                "bullet_ids": ["techcorp_b1", "fake_bullet"],
            },
            {"experience_id": "fake_exp", "bullet_ids": ["x"]},
        ],
        "projects": [{"project_id": "fake_proj", "bullet_ids": ["y"]}],
        "education_ids": ["edu_bsc", "fake_edu"],
        "summary_id": None,
    }
    monkeypatch.setattr("core._call_llm_json", _fake_llm_factory(payload))

    sel = select_cv_content("job", demo_cv, api_key="dummy")

    assert "fake_skill" not in sel.skill_ids
    assert all(e.experience_id != "fake_exp" for e in sel.experiences)
    assert all("fake_bullet" not in e.bullet_ids for e in sel.experiences)
    assert all(p.project_id != "fake_proj" for p in sel.projects)
    assert "fake_edu" not in sel.education_ids


def test_selection_chooses_summary_id(monkeypatch):
    cv = MasterCV(
        personal_info={"name": "A", "email": "a@b.c", "linkedin": "l", "github": "g"},
        facts={"years_experience_total": 5},
        skills=[Skill(id="sk_python", en="Python", fr="Python")],
        summaries=[
            SummaryVariant(id="sum_a", en="A summary.", fr="Résumé A."),
            SummaryVariant(id="sum_b", en="B summary.", fr="Résumé B."),
        ],
    )
    payload = {
        "language": "en",
        "target_role": "Dev",
        "skill_ids": ["sk_python"],
        "experiences": [],
        "projects": [],
        "education_ids": [],
        "summary_id": "sum_b",
    }
    monkeypatch.setattr("core._call_llm_json", _fake_llm_factory(payload))

    sel = select_cv_content("job", cv, api_key="dummy")
    assert sel.summary_id == "sum_b"


def test_selection_summary_falls_back_to_first(monkeypatch):
    cv = MasterCV(
        personal_info={"name": "A", "email": "a@b.c", "linkedin": "l", "github": "g"},
        facts={"years_experience_total": 5},
        skills=[Skill(id="sk_python", en="Python", fr="Python")],
        summaries=[SummaryVariant(id="sum_default", en="S.", fr="R.")],
    )
    payload = {
        "language": "en",
        "target_role": "Dev",
        "skill_ids": ["sk_python"],
        "experiences": [],
        "projects": [],
        "education_ids": [],
        "summary_id": "nonexistent",
    }
    monkeypatch.setattr("core._call_llm_json", _fake_llm_factory(payload))
    sel = select_cv_content("job", cv, api_key="dummy")
    assert sel.summary_id == "sum_default"


# ---------------------------------------------------------------------------
# Summary normalization
# ---------------------------------------------------------------------------


def test_strip_languages_from_summary():
    text = "Senior backend engineer with 6 years. Languages: English (C1), French (B2)."
    cleaned = _strip_languages_from_summary(text)
    assert "Languages" not in cleaned
    assert "Senior backend engineer" in cleaned


def test_normalize_summaries_assigns_default_id():
    cv = MasterCV(
        personal_info={"name": "A", "email": "a@b.c", "linkedin": "l", "github": "g"},
        facts={"years_experience_total": 5},
        summaries=[SummaryVariant(id="whatever", en="Hi.", fr="Salut.")],
    )
    out = _normalize_summaries(cv)
    assert out.summaries[0].id == "sum_default"


def test_normalize_summaries_drops_empty():
    cv = MasterCV(
        personal_info={"name": "A", "email": "a@b.c", "linkedin": "l", "github": "g"},
        facts={"years_experience_total": 5},
        summaries=[SummaryVariant(id="sum_default", en="", fr="")],
    )
    out = _normalize_summaries(cv)
    assert out.summaries == []


# ---------------------------------------------------------------------------
# Categorized skills rendering
# ---------------------------------------------------------------------------


def _categorized_cv() -> MasterCV:
    return MasterCV(
        personal_info={"name": "A", "email": "a@b.c", "linkedin": "l", "github": "g"},
        facts={"years_experience_total": 5},
        skills=[
            Skill(
                id="sk_python",
                en="Python",
                fr="Python",
                category=BilingualText(en="Python & ML", fr="Python & ML"),
            ),
            Skill(
                id="sk_sql",
                en="SQL",
                fr="SQL",
                category=BilingualText(
                    en="Languages & Databases", fr="Langages & Bases de données"
                ),
            ),
            Skill(id="sk_excel", en="Excel", fr="Excel"),  # uncategorized
        ],
    )


def test_skills_grouped_by_category():
    cv = _categorized_cv()
    sel = CVSelection(
        language="en",
        target_role="Dev",
        skill_ids=["sk_sql", "sk_python", "sk_excel"],
        experiences=[],
    )
    tex = latex_builder.render_cv_latex(sel, cv)
    # Both category headers appear.
    assert "Languages" in tex and "Databases" in tex
    assert "Python" in tex and "ML" in tex
    # Uncategorized fallback still renders the bare skill.
    assert "Excel" in tex
    # Category whose best-ranked skill is higher (sk_sql rank 0) comes first.
    assert tex.find("Databases") < tex.find("ML")


def test_skills_fallback_uncategorized_only(demo_cv):
    """Demo CV has no categories -> flat comma-separated line still renders."""
    sel = CVSelection(
        language="en",
        target_role="Backend Engineer",
        skill_ids=[s.id for s in demo_cv.skills],
        experiences=[],
    )
    tex = latex_builder.render_cv_latex(sel, demo_cv)
    assert "Python" in tex and "FastAPI" in tex and "Docker" in tex


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------


def test_summary_rendered_and_localized():
    cv = MasterCV(
        personal_info={"name": "A", "email": "a@b.c", "linkedin": "l", "github": "g"},
        facts={"years_experience_total": 5},
        summaries=[
            SummaryVariant(id="sum_default", en="English summary.", fr="Résumé français.")
        ],
    )
    sel_en = CVSelection(
        language="en", target_role="Dev", skill_ids=[], experiences=[], summary_id="sum_default"
    )
    tex_en = latex_builder.render_cv_latex(sel_en, cv)
    assert "Summary" in tex_en and "English summary." in tex_en

    sel_fr = CVSelection(
        language="fr", target_role="Dev", skill_ids=[], experiences=[], summary_id="sum_default"
    )
    tex_fr = latex_builder.render_cv_latex(sel_fr, cv)
    assert "Profil" in tex_fr and "Résumé français." in tex_fr


def test_summary_absent_when_no_summaries(demo_cv):
    sel = CVSelection(language="en", target_role="Dev", skill_ids=[], experiences=[])
    tex = latex_builder.render_cv_latex(sel, demo_cv)
    assert "summary" not in tex.lower()


# ---------------------------------------------------------------------------
# Layout presets & two-stage overflow loop
# ---------------------------------------------------------------------------


def test_layout_presets_font_floor():
    sizes = [latex_builder.LAYOUT_PRESETS[0]["font_size"], latex_builder.LAYOUT_PRESETS[-1]["font_size"]]
    assert sizes[0] == "10.5pt"
    assert sizes[1] == "9.5pt"
    # Floor constant matches smallest preset.
    assert latex_builder.MIN_FONT_PT == 9.5


def test_render_respects_preset_font_size(demo_cv):
    sel = CVSelection(
        language="en",
        target_role="Dev",
        skill_ids=[s.id for s in demo_cv.skills],
        experiences=[
            SelectedExperience(experience_id="exp_techcorp", bullet_ids=["techcorp_b1"])
        ],
    )
    tex = latex_builder.render_cv_latex(sel, demo_cv, preset=latex_builder.LAYOUT_PRESETS[1])
    assert "10pt" in tex


def test_compile_to_single_page_stage1_compaction(monkeypatch, demo_cv):
    """If the first preset overflows but a tighter one fits, no pruning occurs."""
    rendered_selections = []

    def fake_render(selection, master_cv, preset=None):
        rendered_selections.append(preset["font_size"])
        return "TEX"

    page_counts = {"10.5pt": 2, "10pt": 1}

    def fake_compile(tex, output_dir="output", job_name="cv"):
        # The last-rendered preset decides the page count.
        return "out.pdf", page_counts.get(rendered_selections[-1], 1)

    monkeypatch.setattr(latex_builder, "render_cv_latex", fake_render)
    monkeypatch.setattr(latex_builder, "compile_latex", fake_compile)

    sel = CVSelection(
        language="en",
        target_role="Dev",
        skill_ids=["sk_python"],
        experiences=[SelectedExperience(experience_id="exp_techcorp", bullet_ids=["techcorp_b1", "techcorp_b2"])],
        projects=[SelectedProject(project_id="proj_cvtailor", bullet_ids=["cvtailor_b1"])],
    )
    result = latex_builder.compile_to_single_page(sel, demo_cv)
    assert result == "out.pdf"
    # Stopped at the tight preset; never reached the 9.5pt floor.
    assert rendered_selections == ["10.5pt", "10pt"]


def test_compile_to_single_page_stage2_pruning(monkeypatch, demo_cv):
    """Overflow persists through all presets -> Stage 2 prunes lowest-rank items."""
    render_calls = []

    def fake_render(selection, master_cv, preset=None):
        # Snapshot total bullet count to observe pruning progress.
        total = sum(len(e.bullet_ids) for e in selection.experiences)
        total += sum(len(p.bullet_ids) for p in selection.projects)
        render_calls.append((preset["name"], total))
        return "TEX"

    # Always 2 pages until only the project is gone AND one bullet pruned.
    def fake_compile(tex, output_dir="output", job_name="cv"):
        # Fits only once project removed and at least one exp bullet pruned.
        name, total = render_calls[-1]
        if name == "compact" and total <= 3:
            return "out.pdf", 1
        return "out.pdf", 2

    monkeypatch.setattr(latex_builder, "render_cv_latex", fake_render)
    monkeypatch.setattr(latex_builder, "compile_latex", fake_compile)

    sel = CVSelection(
        language="en",
        target_role="Dev",
        skill_ids=["sk_python"],
        experiences=[SelectedExperience(experience_id="exp_techcorp", bullet_ids=["techcorp_b1", "techcorp_b2"])],
        projects=[SelectedProject(project_id="proj_cvtailor", bullet_ids=["cvtailor_b1"])],
    )
    # total initially = 2 exp + 1 project = 3 -> already <=3, project at back has 1
    # bullet so it gets dropped (total->2) then fits.
    result = latex_builder.compile_to_single_page(sel, demo_cv)
    assert result == "out.pdf"
    # Original selection must not be mutated.
    assert len(sel.projects) == 1
    assert len(sel.experiences[0].bullet_ids) == 2


def test_prune_one_content_item_prefers_lowest_project(demo_cv):
    sel = CVSelection(
        language="en",
        target_role="Dev",
        skill_ids=["sk_python"],
        experiences=[
            SelectedExperience(
                experience_id="exp_techcorp",
                bullet_ids=["techcorp_b1", "techcorp_b2"],
            )
        ],
        projects=[
            SelectedProject(project_id="proj_cvtailor", bullet_ids=["cvtailor_b1"])
        ],
    )
    pruned, desc = latex_builder._prune_one_content_item(sel)
    # Project has only 1 bullet -> dropped entirely.
    assert pruned.projects == []
    assert desc is not None and "proj_cvtailor" in desc
    # Experience untouched.
    assert pruned.experiences[0].bullet_ids == ["techcorp_b1", "techcorp_b2"]
    # Original untouched.
    assert len(sel.projects) == 1
