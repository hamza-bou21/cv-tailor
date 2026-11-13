import pytest
from core import (
    load_master_cv,
    verify_skill_claims,
    verify_citation_ids,
    verify_numerals,
    _extract_claimed_skills,
)
from models import MasterCV


@pytest.fixture
def demo_cv():
    return load_master_cv("demo/master_cv.yaml")


def test_catches_unsupported_skill(demo_cv):
    """AC-3b: Catch skills claimed by the model not present in Master CV."""
    claimed_skills = ["Python", "Docker", "Kubernetes"]  # Kubernetes is not in demo CV
    invalid = verify_skill_claims(claimed_skills, demo_cv)
    assert "Kubernetes" in invalid
    assert "Python" not in invalid


def test_catches_fabricated_citation_id(demo_cv):
    """AC-3c: Catch citations pointing to non-existent bullet IDs."""
    cited_ids = ["techcorp_b1", "fake_bullet_id_999"]
    invalid = verify_citation_ids(cited_ids, demo_cv)
    assert "fake_bullet_id_999" in invalid
    assert "techcorp_b1" not in invalid


def test_catches_fabricated_numeral(demo_cv):
    """AC-3a: Catch fabricated numerical metrics."""
    # Allowed: 8000 (in techcorp_b1 & facts), 1 (allowlist)
    # Fabricated: 50000, 15
    sentence = "I led 15 engineers managing 50000 requests across 1 cluster."
    invalid = verify_numerals(sentence, demo_cv)
    assert "15" in invalid
    assert "50000" in invalid
    assert "1" not in invalid


# ---------------------------------------------------------------------------
# Issue 1: grouped-skill verification and single-letter "R" handling
# ---------------------------------------------------------------------------


def _grouped_skill_cv() -> MasterCV:
    """A MasterCV whose YAML still stores a grouped skill (older/manual profile)."""
    return MasterCV(
        personal_info={
            "name": "Data Analyst",
            "email": "da@example.com",
            "linkedin": "linkedin.com/in/da",
            "github": "github.com/da",
        },
        facts={"years_experience_total": 3},
        skills=[
            {
                "id": "sk_python",
                "en": "Python (Pandas, NumPy, Scikit-learn)",
                "fr": "Python (Pandas, NumPy, Scikit-learn)",
            },
            {"id": "sk_sql", "en": "SQL", "fr": "SQL"},
        ],
    )


def test_grouped_yaml_skills_do_not_trigger_false_positive():
    """Atomic skills claimed in prose pass even when YAML stores them grouped."""
    cv = _grouped_skill_cv()
    claimed = ["Python", "Pandas", "NumPy", "Scikit-learn", "SQL"]
    invalid = verify_skill_claims(claimed, cv)
    assert invalid == [], invalid


def test_grouped_yaml_still_rejects_absent_skill():
    cv = _grouped_skill_cv()
    invalid = verify_skill_claims(["Python", "TensorFlow"], cv)
    assert "TensorFlow" in invalid
    assert "Python" not in invalid


def test_single_letter_r_requires_uppercase_standalone():
    """French prose (lowercase 'r') must never register as the 'R' skill."""
    prose = "Je suis rigoureux, réactif et je rédige des rapports réguliers."
    assert "R" not in _extract_claimed_skills(prose)
    # A genuine standalone uppercase 'R' is detected.
    assert "R" in _extract_claimed_skills("Proficient in R and Python.")


def test_single_letter_r_not_matched_inside_words():
    """'Ruby', 'React', 'Redis', 'REST' must not masquerade as the 'R' skill."""
    claimed = _extract_claimed_skills("Ruby, React, Redis, REST APIs")
    assert "R" not in claimed
    assert {"Ruby", "React", "Redis", "REST"} <= set(claimed)
