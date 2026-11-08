import os
import json
import yaml
from dotenv import load_dotenv
from groq import Groq
from models import MasterCV, FitAnalysis, CVSelection, SelectedExperience, SelectedProject
import re
from typing import Literal

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
    
    # 1. Format facts metrics
    facts_lines = []
    for k, v in master_cv.facts.model_dump().items():
        if v is not None:
            clean_name = k.replace("_", " ").title()
            facts_lines.append(f"  • {clean_name}: {v}")
    facts_summary = "\n".join(facts_lines)

    # 2. Format skills
    known_skills = [s.en for s in master_cv.skills]

    # 3. Format experience with bullet highlights
    exp_sections = []
    for exp in master_cv.experience:
        bullets = "\n".join(f"    - {b.en}" for b in exp.bullets)
        exp_sections.append(
            f"  • {exp.role.en} at {exp.company} ({exp.start_date} to {exp.end_date}):\n{bullets}"
        )
    experience_summary = "\n".join(exp_sections)
    
    prompt = f"""
    You are an objective technical recruiter assessing fit between a candidate and a job opening.
    
    CANDIDATE PROFILE:
    Documented Facts:
{facts_summary}

    Documented Skills:
    {', '.join(known_skills)}

    Documented Experience & Verified Achievements:
{experience_summary}
    
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
def select_cv_content(
    job_text: str,
    master_cv: MasterCV,
    language: Literal["en", "fr"] = "en",
    model: str = "openai/gpt-oss-120b"
) -> CVSelection:
    """
    Selects and ranks existing CV items by ID to match the target job posting (FR-8).
    Enforces deterministic validation so any hallucinated IDs are dropped.
    """
    schema = CVSelection.model_json_schema()

    # Build a structured catalog containing only IDs and text in the requested language
    catalog = {
        "skills": [{"id": s.id, "name": getattr(s, language)} for s in master_cv.skills],
        "experience": [
            {
                "id": exp.id,
                "role": getattr(exp.role, language),
                "company": exp.company,
                "bullets": [{"id": b.id, "text": getattr(b, language)} for b in exp.bullets]
            }
            for exp in master_cv.experience
        ],
        "projects": [
            {
                "id": proj.id,
                "name": proj.name,
                "description": getattr(proj.description, language),
                "bullets": [{"id": b.id, "text": getattr(b, language)} for b in proj.bullets]
            }
            for proj in master_cv.projects
        ],
        "education": [
            {
                "id": edu.id,
                "degree": getattr(edu.degree, language),
                "institution": edu.institution
            }
            for edu in master_cv.education
        ]
    }

    prompt = f"""
    You are an expert CV tailoring engine. Your task is to select and prioritize the most relevant items 
    from the candidate's CV catalog for the provided job posting.

    TARGET LANGUAGE: {language}

    CANDIDATE CATALOG:
    {json.dumps(catalog, indent=2)}

    JOB DESCRIPTION:
    \"\"\"{job_text}\"\"\"

    SELECTION RULES:
    1. STRICT PROVENANCE: Use ONLY IDs that exist in the candidate catalog. Do NOT fabricate IDs.
    2. Rank skill_ids strictly from most relevant to least relevant.
    3. For each experience, select only bullets that demonstrate relevant impact, ranked by relevance.
    4. Select up to 2-3 most relevant projects.
    5. target_role must be a concise title adapted to the job description (e.g. 'Senior Backend Engineer').
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"You are a strict data selector. Output JSON matching schema: {json.dumps(schema)}"},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    data = json.loads(response.choices[0].message.content)
    raw_selection = CVSelection(**data)

    # Deterministic scrubbing: drop any IDs not present in master_cv
    valid_skill_ids = {s.id for s in master_cv.skills}
    valid_exp_ids = {exp.id: {b.id for b in exp.bullets} for exp in master_cv.experience}
    valid_proj_ids = {proj.id: {b.id for b in proj.bullets} for proj in master_cv.projects}
    valid_edu_ids = {edu.id for edu in master_cv.education}

    cleaned_skills = [sid for sid in raw_selection.skill_ids if sid in valid_skill_ids]

    cleaned_experiences = []
    for exp in raw_selection.experiences:
        if exp.experience_id in valid_exp_ids:
            allowed_bullets = valid_exp_ids[exp.experience_id]
            valid_bullets = [bid for bid in exp.bullet_ids if bid in allowed_bullets]
            cleaned_experiences.append(
                SelectedExperience(experience_id=exp.experience_id, bullet_ids=valid_bullets)
            )

    cleaned_projects = []
    for proj in raw_selection.projects:
        if proj.project_id in valid_proj_ids:
            allowed_bullets = valid_proj_ids[proj.project_id]
            valid_bullets = [bid for bid in proj.bullet_ids if bid in allowed_bullets]
            cleaned_projects.append(
                SelectedProject(project_id=proj.project_id, bullet_ids=valid_bullets)
            )

    cleaned_education = [eid for eid in raw_selection.education_ids if eid in valid_edu_ids]

    return CVSelection(
        language=raw_selection.language,
        target_role=raw_selection.target_role,
        skill_ids=cleaned_skills,
        experiences=cleaned_experiences,
        projects=cleaned_projects,
        education_ids=cleaned_education
    )