import os
import json
import time
import re
from typing import Literal
import yaml
from dotenv import load_dotenv
from groq import Groq
import pymupdf

from models import (
    MasterCV, 
    FitAnalysis, 
    CVSelection, 
    SelectedExperience, 
    SelectedProject,
    CoverLetter, 
    CoverLetterDraft,
    CoverLetterSentence
)

load_dotenv()

ALLOWLISTED_NUMERALS = {"one", "two", "three", "first", "second", "1", "2"}


def _call_llm_json(
    messages: list[dict],
    schema: dict,
    api_key: str,
    model: str,
    temperature: float = 0.1,
    max_attempts: int = 3,
) -> dict:
    """Calls Groq with retry/backoff and robust JSON parsing.

    A fresh client is initialized per call from the caller-supplied ``api_key`` so
    no key is ever baked into the module or shared across requests.
    Retries on transient API errors and on malformed JSON payloads, raising a
    clear RuntimeError if all attempts fail.
    """
    if not api_key or not api_key.strip():
        raise ValueError("A Groq API key is required to call the language model.")

    client = Groq(api_key=api_key)
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except json.JSONDecodeError as exc:
            last_error = exc
            print(f"[!] Malformed JSON from LLM (attempt {attempt}/{max_attempts}).")
        except Exception as exc:  # network / rate-limit / API errors
            last_error = exc
            print(f"[!] LLM call failed (attempt {attempt}/{max_attempts}): {exc}")

        if attempt < max_attempts:
            time.sleep(min(2 ** (attempt - 1), 8))

    raise RuntimeError(f"LLM call failed after {max_attempts} attempts: {last_error}")


def load_master_cv(yaml_path: str = "master_cv.yaml") -> MasterCV:
    """Loads and validates the local Master CV YAML."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Master CV file not found at: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return MasterCV(**data)


def analyze_job(job_text: str, master_cv: MasterCV, api_key: str, model: str = "openai/gpt-oss-120b") -> FitAnalysis:
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

    response_dict = _call_llm_json(
        messages=[
            {"role": "system", "content": f"You are a strict evaluation engine. Output JSON matching schema: {json.dumps(schema)}"},
            {"role": "user", "content": prompt}
        ],
        schema=schema,
        api_key=api_key,
        model=model,
    )
    return FitAnalysis(**response_dict)


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


_NUMERAL_RE = re.compile(r'\d+(?:,\d+)*(?:\.\d+)?')


def _extract_numerals(text: str) -> set[str]:
    """Extracts numeric tokens and adds normalized variants (comma-free, integer part).

    This tolerates formatting differences such as '1,000' vs '1000' or '30.0' vs '30'
    so equivalent numbers are treated as the same value during verification.
    """
    variants: set[str] = set()
    for token in _NUMERAL_RE.findall(text or ""):
        variants.add(token)
        comma_stripped = token.replace(",", "")
        variants.add(comma_stripped)
        # Integer part only, e.g. '30.0' -> '30' and '99.8' -> '99'
        if "." in comma_stripped:
            variants.add(comma_stripped.split(".")[0])
    return variants


def verify_numerals(text: str, master_cv: MasterCV) -> list[str]:
    """FR-10c: Deterministic regex check for numbers outside facts and allowlist."""
    found_numbers = _NUMERAL_RE.findall(text)

    valid_numbers = {
        str(v) for v in master_cv.facts.model_dump().values() if v is not None
    } | ALLOWLISTED_NUMERALS

    # Allow numbers appearing naturally in experience bullets (EN + FR)
    for exp in master_cv.experience:
        for b in exp.bullets:
            valid_numbers.update(_extract_numerals(b.en))
            valid_numbers.update(_extract_numerals(b.fr))

    # Allow numbers appearing naturally in project bullets (EN + FR)
    for proj in master_cv.projects:
        for b in proj.bullets:
            valid_numbers.update(_extract_numerals(b.en))
            valid_numbers.update(_extract_numerals(b.fr))

    # Normalize the candidate tokens too, so formatting differences don't cause false flags
    return [num for num in found_numbers if not (_extract_numerals(num) & valid_numbers)]


def select_cv_content(
    job_text: str,
    master_cv: MasterCV,
    api_key: str,
    language: Literal["en", "fr"] = "en",
    model: str = "openai/gpt-oss-120b"
) -> CVSelection:
    """
    Selects and ranks existing CV items by ID to match the target job posting (FR-8).
    Enforces deterministic validation so any hallucinated IDs are dropped.
    """
    schema = CVSelection.model_json_schema()

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

    data = _call_llm_json(
        messages=[
            {"role": "system", "content": f"You are a strict data selector. Output JSON matching schema: {json.dumps(schema)}"},
            {"role": "user", "content": prompt}
        ],
        schema=schema,
        api_key=api_key,
        model=model,
    )
    raw_selection = CVSelection(**data)

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

    # Safety net: never ship a CV with zero experience. If the LLM selection was
    # emptied by validation, fall back to every experience (with all its bullets).
    if not cleaned_experiences and master_cv.experience:
        print("[!] No experiences survived validation; falling back to all experiences.")
        cleaned_experiences = [
            SelectedExperience(experience_id=exp.id, bullet_ids=[b.id for b in exp.bullets])
            for exp in master_cv.experience
        ]

    return CVSelection(
        language=language,
        target_role=raw_selection.target_role,
        skill_ids=cleaned_skills,
        experiences=cleaned_experiences,
        projects=cleaned_projects,
        education_ids=cleaned_education
    )


def verify_cover_letter(letter: CoverLetterDraft, master_cv: MasterCV) -> list[str]:
    """Runs deterministic verification on citations and numerals across all sentences."""
    errors = []

    # 1. Check citation IDs
    all_cited_ids = [cid for s in letter.sentences for cid in s.based_on]
    invalid_ids = verify_citation_ids(all_cited_ids, master_cv)
    if invalid_ids:
        errors.append(f"Invalid citation IDs referenced: {', '.join(invalid_ids)}")

    # 2. Check numerals across sentence text
    full_text = " ".join(s.text for s in letter.sentences)
    invalid_numerals = verify_numerals(full_text, master_cv)
    if invalid_numerals:
        errors.append(f"Fabricated numerals outside facts/allowlist: {', '.join(invalid_numerals)}")

    return errors


def generate_cover_letter(
    job_text: str,
    master_cv: MasterCV,
    api_key: str,
    language: Literal["en", "fr"] = "en",
    model: str = "openai/gpt-oss-120b",
    max_retries: int = 2
) -> CoverLetter:
    """
    FR-9 & FR-10: Generates a tailored 3-4 paragraph cover letter where every factual claim
    cites a Master CV ID. Automatically triggers re-prompting if verification checks fail.
    """
    schema = CoverLetterDraft.model_json_schema()

    catalog = {
        "candidate_name": master_cv.personal_info.name,
        "facts": master_cv.facts.model_dump(),
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
                "bullets": [{"id": b.id, "text": getattr(b, language)} for b in proj.bullets]
            }
            for proj in master_cv.projects
        ]
    }

    base_prompt = f"""
You are an expert technical cover letter writer.
Draft a professional, compelling, 3-to-4 paragraph cover letter tailored to the job description.

TARGET LANGUAGE: {language}

CANDIDATE DATA & VALID IDs:
{json.dumps(catalog, indent=2)}

JOB DESCRIPTION:
\"\"\"{job_text}\"\"\"

CRITICAL HONESTY & FORMAT RULES:
1. Provide the letter as a flat array of 'sentences'. Tag each sentence with its 'paragraph' number (1, 2, 3, or 4).
2. Every sentence making a factual statement (experience, metrics, tooling) MUST populate 'based_on' with valid IDs from candidate data.
3. If a sentence is an opening greeting, transition, or closing signoff, 'based_on' must be an empty list [].
4. DO NOT invent metrics or numbers. Only use numbers present in the candidate facts or bullets.
5. DO NOT reference technologies or skills not listed in the candidate data.
"""

    messages = [
        {"role": "system", "content": f"You are a strict technical writer with zero-hallucination constraints. Output JSON matching schema: {json.dumps(schema)}"},
        {"role": "user", "content": base_prompt}
    ]

    for attempt in range(max_retries + 1):
        raw_content = None
        data = _call_llm_json(
            messages=messages,
            schema=schema,
            api_key=api_key,
            model=model,
        )
        # Keep a serialized copy so we can echo it back as the assistant turn on retries.
        raw_content = json.dumps(data)

        draft = CoverLetterDraft(**data)

        errors = verify_cover_letter(draft, master_cv)

        if not errors:
            return CoverLetter(**draft.model_dump(), verification_passed=True, verification_warnings=[])

        print(f"[!] Verification failed on attempt {attempt + 1}: {errors}")

        if attempt < max_retries:
            correction_instruction = f"""
Your previous output failed automated fact-checking checks with the following violations:
{json.dumps(errors, indent=2)}

Rewrite the cover letter to completely resolve these violations.
Ensure every cited ID strictly matches an ID from the candidate catalog, and remove or correct any fabricated numerals.
"""
            messages.append({"role": "assistant", "content": raw_content})
            messages.append({"role": "user", "content": correction_instruction})

    return CoverLetter(**draft.model_dump(), verification_passed=False, verification_warnings=errors)


def extract_text_from_pdf(pdf_source: bytes | str) -> str:
    """Extracts raw text from a PDF file path or raw bytes buffer."""
    if isinstance(pdf_source, (bytes, bytearray)):
        doc = pymupdf.open(stream=pdf_source, filetype="pdf")
    else:
        doc = pymupdf.open(pdf_source)
    
    extracted_text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return extracted_text.strip()


def parse_pdf_to_master_cv(
    pdf_text: str, 
    api_key: str,
    model: str = "openai/gpt-oss-120b"
) -> MasterCV:
    """
    Parses unstructured CV text into a verified MasterCV object.
    Generates unique ID slugs for every item and provides bilingual EN/FR fields.
    """
    schema = MasterCV.model_json_schema()

    prompt = f"""
You are an expert ATS data engineer. Convert this unstructured resume into a Master CV YAML definition.

RESUME TEXT:
\"\"\"{pdf_text}\"\"\"

EXTRACTION REQUIREMENTS:
1. PERSONAL INFO: Name, email, phone, location, linkedin, github.
2. SKILLS: Every single technical tool, language, or framework with id 'sk_<name>'.
3. EXPERIENCE: You MUST extract all work experience entries. For each role, provide 1-3 achievement bullets with realistic French ('fr') translations alongside English ('en').
4. PROJECTS: If the candidate lists academic or personal projects, extract them under 'projects'.
5. EDUCATION: You MUST extract degree, institution name, and graduation year under 'education'.
6. Do NOT leave 'experience' or 'education' empty if there is any mention of work, study, or degrees in the text.
"""

    data = _call_llm_json(
        messages=[
            {"role": "system", "content": f"You are a strict data transformation engine. Output JSON matching schema: {json.dumps(schema)}"},
            {"role": "user", "content": prompt}
        ],
        schema=schema,
        api_key=api_key,
        model=model,
    )
    return MasterCV(**data)


def save_master_cv(cv: MasterCV, target_path: str):
    """Saves a MasterCV instance to a formatted YAML file (null fields omitted)."""
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        yaml.dump(
            cv.model_dump(exclude_none=True),
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )