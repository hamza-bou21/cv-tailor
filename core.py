import os
import json
import yaml
from dotenv import load_dotenv
from groq import Groq
from models import MasterCV, FitAnalysis
import re

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def load_master_cv(yaml_path: str = "master_cv.yaml") -> MasterCV:
    """Loads and validates the local Master CV YAML."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Master CV file not found at: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return MasterCV(**data)

def analyze_job(job_text: str, master_cv: MasterCV, model: str = "openai/gpt-oss-120b") -> FitAnalysis:
    """
    Performs the fit analysis between the job description and the Master CV.
    Enforces FR-6: Validates against FitAnalysis schema and discards invalid data.
    """
    schema = FitAnalysis.model_json_schema()
    
    # Minimize token usage by preparing a lean summary of candidate facts & skills
    known_skills = [s.en for s in master_cv.skills]
    roles_summary = [
        f"{exp.role.en} at {exp.company} ({exp.start_date} to {exp.end_date})"
        for exp in master_cv.experience
    ]
    
    prompt = f"""
    You are an objective technical recruiter assessing fit between a candidate and a job opening.
    
    CANDIDATE PROFILE:
    - Total Years Experience: {master_cv.facts.years_experience_total}
    - Documented Skills: {', '.join(known_skills)}
    - Past Roles: {'; '.join(roles_summary)}
    
    JOB DESCRIPTION:
    \"\"\"{job_text}\"\"\"
    
    RULES:
    1. Score fit strictly from 0 to 100 based on core requirements met.
    2. Verdict must be:
       - 'apply' if score >= 75
       - 'stretch' if score >= 50 and < 75
       - 'skip' if score < 50
    3. 'strengths' must only reference verified candidate capabilities.
    4. 'gaps' must list unmet job requirements.
    5. 'covered_keywords' must only include keywords the candidate demonstrably has.
    6. 'missing_keywords' must list keywords present in the job but absent from candidate's profile.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"You are a strict evaluation engine. Output JSON matching schema: {json.dumps(schema)}"},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    
    result_dict = json.loads(response.choices[0].message.content)
    analysis = FitAnalysis(**result_dict)
    return analysis
ALLOWLISTED_NUMERALS = {"one", "two", "three", "first", "second", "1", "2"}

def verify_skill_claims(claimed_skills: list[str], master_cv: MasterCV) -> list[str]:
    """FR-10a: Deterministic set check. Returns any unapproved skills."""
    valid_names = {s.en.lower() for s in master_cv.skills} | {s.fr.lower() for s in master_cv.skills}
    return [s for s in claimed_skills if s.lower() not in valid_names]

def verify_citation_ids(cited_ids: list[str], master_cv: MasterCV) -> list[str]:
    """FR-10b: Validates that cited IDs actually exist in the Master CV."""
    all_valid_ids = {s.id for s in master_cv.skills}
    for exp in master_cv.experience:
        all_valid_ids.add(exp.id)
        all_valid_ids.update(b.id for b in exp.bullets)
    for proj in master_cv.projects:
        all_valid_ids.add(proj.id)
        all_valid_ids.update(b.id for b in proj.bullets)
    for edu in master_cv.education:
        all_valid_ids.add(edu.id)

    return [cid for cid in cited_ids if cid not in all_valid_ids]

def verify_numerals(text: str, master_cv: MasterCV) -> list[str]:
    """FR-10c: Deterministic regex check for numbers outside facts and allowlist."""
    found_numbers = re.findall(r'\b\d+(?:,\d+)?(?:\.\d+)?\b', text)
    valid_numbers = {
        str(v) for v in master_cv.facts.model_dump().values() if v is not None
    } | ALLOWLISTED_NUMERALS

    # Also allow numbers that already appear naturally in your CV text
    for exp in master_cv.experience:
        for b in exp.bullets:
            valid_numbers.update(re.findall(r'\b\d+(?:,\d+)?(?:\.\d+)?\b', b.en))
            valid_numbers.update(re.findall(r'\b\d+(?:,\d+)?(?:\.\d+)?\b', b.fr))

    return [num for num in found_numbers if num not in valid_numbers]