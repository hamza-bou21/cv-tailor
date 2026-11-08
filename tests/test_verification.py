import pytest
from core import load_master_cv, verify_skill_claims, verify_citation_ids, verify_numerals

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