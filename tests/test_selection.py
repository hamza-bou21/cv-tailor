import pytest
from core import load_master_cv, verify_citation_ids
from models import SelectedExperience, CVSelection

@pytest.fixture
def demo_cv():
    return load_master_cv("demo/master_cv.yaml")

def test_deterministic_id_scrubbing(demo_cv):
    """Verify that hallucinated IDs injected into CVSelection are detected."""
    invalid_ids = ["fake_skill_99", "fake_exp_00", "fake_bullet_11"]
    detected = verify_citation_ids(invalid_ids, demo_cv)
    assert set(detected) == set(invalid_ids)