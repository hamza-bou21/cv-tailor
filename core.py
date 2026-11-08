import os
import json
import yaml
from dotenv import load_dotenv
from groq import Groq
from models import MasterCV, FitAnalysis

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