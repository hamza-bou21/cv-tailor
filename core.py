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
    CoverLetterSentence,
    PdfLink,
    Skill,
    SummaryVariant,
)

load_dotenv()

ALLOWLISTED_NUMERALS = {"one", "two", "three", "first", "second", "1", "2"}

# Reference vocabulary of common technical skills/tools. Used to detect skills a
# generated cover letter *claims* in prose so we can flag any that are NOT in the
# candidate's Master CV (FR-10a honesty enforcement). This is intentionally not
# exhaustive: it only needs to cover terms a model is likely to mention when
# over-claiming. Matching is case-insensitive and word-boundary aware.
COMMON_TECH_SKILLS = {
    # Languages
    "Python",
    "Java",
    "JavaScript",
    "TypeScript",
    "Go",
    "Golang",
    "Rust",
    "C++",
    "C#",
    "Ruby",
    "PHP",
    "Scala",
    "Kotlin",
    "Swift",
    "R",
    "MATLAB",
    "Perl",
    "Elixir",
    "Haskell",
    "Clojure",
    "Dart",
    "Lua",
    "Bash",
    "Shell",
    "SQL",
    "HTML",
    "CSS",
    # Frameworks / libraries
    "FastAPI",
    "Flask",
    "Django",
    "Spring",
    "Spring Boot",
    "Express",
    "NestJS",
    "Node.js",
    "Node",
    "React",
    "Vue",
    "Angular",
    "Next.js",
    "Svelte",
    "Rails",
    "Ruby on Rails",
    ".NET",
    "ASP.NET",
    "Laravel",
    "GraphQL",
    "gRPC",
    "REST",
    "Pandas",
    "NumPy",
    "PyTorch",
    "TensorFlow",
    "scikit-learn",
    "Keras",
    "Spark",
    "Airflow",
    "dbt",
    "Kafka",
    "RabbitMQ",
    "Celery",
    "Hibernate",
    # Data stores
    "PostgreSQL",
    "MySQL",
    "SQLite",
    "MongoDB",
    "Redis",
    "Cassandra",
    "DynamoDB",
    "Elasticsearch",
    "ClickHouse",
    "Snowflake",
    "BigQuery",
    "Oracle",
    "MariaDB",
    "Neo4j",
    "Memcached",
    # Cloud / infra / devops
    "AWS",
    "GCP",
    "Azure",
    "Google Cloud",
    "Docker",
    "Kubernetes",
    "Terraform",
    "Ansible",
    "Jenkins",
    "GitHub Actions",
    "GitLab CI",
    "CircleCI",
    "Helm",
    "Prometheus",
    "Grafana",
    "Nginx",
    "Linux",
    "Unix",
    "Vagrant",
    "Pulumi",
    "CloudFormation",
    "Serverless",
    "Lambda",
    "EC2",
    "S3",
    "ECS",
    "EKS",
    # Practices / concepts
    "CI/CD",
    "TDD",
    "Agile",
    "Scrum",
    "Kanban",
    "Microservices",
    "Monolith",
    "Machine Learning",
    "Deep Learning",
    "NLP",
    "Computer Vision",
    "ETL",
    "ELT",
    "Data Warehouse",
    "Data Lake",
    "Streaming",
    "Observability",
    "DevOps",
    "Verilog",
    "VHDL",
    "FPGA",
    "Embedded",
    "RTOS",
    "PID",
}


def _is_rate_limit_error(exc: Exception) -> bool:
    """True if an exception looks like a Groq rate/token-limit (429/413) error."""
    status = getattr(exc, "status_code", None)
    if status in (413, 429):
        return True
    msg = str(exc).lower()
    return (
        "rate_limit" in msg
        or "rate limit" in msg
        or "error code: 429" in msg
        or "error code: 413" in msg
        or "request too large" in msg
    )


def _reasoning_kwargs_for_model(model: str) -> dict:
    """Returns extra ``chat.completions.create`` kwargs for reasoning models.

    Reasoning models (``gpt-oss``, ``qwen``) spend part of ``max_tokens`` on
    internal reasoning before emitting JSON. On Groq that can exhaust the budget
    and raise "max completion tokens reached before generating a valid document".
    Capping their reasoning effort keeps enough budget for the JSON document.

    Plain models (e.g. ``llama-3.3-70b-versatile``) reject the unknown
    ``reasoning_effort`` parameter, so return an empty dict for them.
    """
    model_lower = (model or "").lower()
    if "gpt-oss" in model_lower:
        return {"reasoning_effort": "low"}
    if "qwen" in model_lower:
        return {"reasoning_effort": "none"}
    return {}


def _call_llm_json(
    messages: list[dict],
    schema: dict,
    api_key: str,
    model: str,
    temperature: float = 0.0,
    max_attempts: int = 3,
    max_tokens: int = 2048,
) -> dict:
    """Calls Groq with retry/backoff and robust JSON parsing.

    A fresh client is initialized per call from the caller-supplied ``api_key`` so
    no key is ever baked into the module or shared across requests.
    Retries on transient API errors and on malformed JSON payloads, raising a
    clear RuntimeError if all attempts fail.

    ``max_tokens`` is a per-call budget. Groq's free tier enforces an 8,000 TPM
    single-request ceiling that sums ``prompt_tokens + max_tokens``; callers must
    right-size it to their prompt (e.g. 2600 for cover letter / selection / fit
    analysis, 6000 for full CV extraction) so the combined total stays under the
    limit while still leaving room for the completion. For reasoning models, the
    reasoning effort is capped (see :func:`_reasoning_kwargs_for_model`) so those
    internal tokens don't consume the whole budget before JSON is emitted.
    """
    if not api_key or not api_key.strip():
        raise ValueError("A Groq API key is required to call the language model.")

    # Reasoning models (gpt-oss, qwen) spend part of ``max_tokens`` on internal
    # reasoning before emitting JSON. On Groq that can exhaust the budget and
    # raise "max completion tokens reached before generating a valid document".
    # Cap their reasoning with ``reasoning_effort`` so the JSON still fits. Plain
    # models (e.g. llama-3.3-70b-versatile) reject this arg, so only send it for
    # known reasoning model families.
    reasoning_kwargs = _reasoning_kwargs_for_model(model)

    client = Groq(api_key=api_key)
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens,
                **reasoning_kwargs,
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except json.JSONDecodeError as exc:
            last_error = exc
            print(f"[!] Malformed JSON from LLM (attempt {attempt}/{max_attempts}).")
        except Exception as exc:  # network / rate-limit / API errors
            last_error = exc
            print(f"[!] LLM call failed (attempt {attempt}/{max_attempts}): {exc}")
            # Rate-limit / token-limit (429 or 413): wait for the rolling 60s TPM
            # window to clear before retrying, otherwise every retry re-trips it.
            if _is_rate_limit_error(exc) and attempt < max_attempts:
                print("[!] Rate/token limit hit; sleeping 10s before retrying...")
                time.sleep(10)

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


def analyze_job(
    job_text: str, master_cv: MasterCV, api_key: str, model: str = "openai/gpt-oss-120b"
) -> FitAnalysis:
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

    # 4. Format projects (name, description, and all bullets) so project-level
    #    technical achievements are evaluated in strengths/gaps/keywords.
    proj_sections = []
    for proj in master_cv.projects:
        bullets = "\n".join(f"    - {b.en}" for b in proj.bullets)
        proj_sections.append(f"  • {proj.name} — {proj.description.en}:\n{bullets}")
    projects_summary = "\n".join(proj_sections)

    # 5. Spoken languages and any other skills live in master_cv.skills already.
    prompt = f"""
    You are an objective technical recruiter assessing fit between a candidate and a job opening.
    
    CANDIDATE PROFILE:
    Documented Facts:
{facts_summary}

    Documented Skills (including spoken languages with proficiency):
    {', '.join(known_skills)}

    Documented Experience & Verified Achievements:
{experience_summary}

    Documented Projects & Technical Achievements:
{projects_summary}
    
    JOB DESCRIPTION:
    \"\"\"{job_text}\"\"\"
    
    RULES:
    1. Score fit strictly from 0 to 100 based on core requirements met.
    2. Verdict must be:
       - 'apply' if score >= 75
       - 'stretch' if score >= 50 and < 75
       - 'skip' if score < 50
    3. 'strengths' must only reference verified candidate capabilities. Consider BOTH
       experience bullets AND project bullets as evidence.
    4. 'gaps' must list unmet job requirements — only list a gap if NO experience or
       project bullet demonstrates the capability.
    5. 'covered_keywords' must only include keywords the candidate demonstrably has
       (from skills, experience, or projects).
    6. 'missing_keywords' must list keywords present in the job but absent from candidate's profile.
    """

    response_dict = _call_llm_json(
        messages=[
            {
                "role": "system",
                "content": f"You are a strict evaluation engine. Output JSON matching schema: {json.dumps(schema)}",
            },
            {"role": "user", "content": prompt},
        ],
        schema=schema,
        api_key=api_key,
        model=model,
        max_tokens=2600,
    )
    return FitAnalysis(**response_dict)


_SKILL_GROUP_RE = re.compile(r"^(?P<base>[^()]+?)\s*\((?P<inner>[^()]*)\)\s*$")

# Parenthetical values that denote a proficiency/level rather than a grouped
# sub-skill (e.g. "English (C1)", "Arabic (native)", "French (B2)").
_PROFICIENCY_TOKENS = {
    "native",
    "natif",
    "native speaker",
    "fluent",
    "courant",
    "bilingual",
    "bilingue",
    "professional",
    "professionnel",
    "intermediate",
    "intermédiaire",
    "basic",
    "débutant",
    "beginner",
    "advanced",
    "avancé",
    "conversational",
    "a1",
    "a2",
    "b1",
    "b2",
    "c1",
    "c2",
    "mother tongue",
    "langue maternelle",
}


def _is_proficiency_parenthetical(inner: str) -> bool:
    """True if a parenthetical denotes a spoken-language level, not sub-skills."""
    norm = inner.strip().lower()
    return norm in _PROFICIENCY_TOKENS


def _split_skill_string(name: str) -> list[str]:
    """Splits a grouped skill label into atomic skill names.

    Examples (only when the parenthetical holds sub-skills, not a proficiency):
      'Python (Pandas, NumPy, Scikit-learn)' -> ['Python', 'Pandas', 'NumPy', 'Scikit-learn']
      'Python/Pandas/NumPy'                  -> ['Python', 'Pandas', 'NumPy']
    Proficiency parentheticals are preserved as-is:
      'English (C1)' -> ['English (C1)']
    """
    if not name:
        return []
    match = _SKILL_GROUP_RE.match(name.strip())
    if match and not _is_proficiency_parenthetical(match.group("inner")):
        base = match.group("base").strip()
        inner = match.group("inner")
        parts = re.split(r"\s*[,/;|]\s*", inner)
        atoms = [base] + [p.strip() for p in parts if p.strip()]
        return [a for a in atoms if a]

    # No parenthetical grouping: split any slash/comma-delimited listing.
    if re.search(r"[,/;|]", name):
        parts = re.split(r"\s*[,/;|]\s*", name)
        return [p.strip() for p in parts if p.strip()]

    return [name.strip()]


def _skill_slug(name: str) -> str:
    """Builds a clean skill id slug, e.g. 'Scikit-learn' -> 'sk_scikit_learn'."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return f"sk_{slug}" if slug else "sk_skill"


def _expand_skill_aliases(name: str) -> set[str]:
    """Returns the lowercase alias set for one skill label (base + sub-skills).

    'Python (Pandas, NumPy, Scikit-learn)' -> {'python', 'pandas', 'numpy',
    'scikit-learn'}. Proficiency parentheticals stay intact ('english (c1)').
    """
    aliases: set[str] = set()
    for atom in _split_skill_string(name):
        aliases.add(atom.lower())
    aliases.add(name.strip().lower())
    return aliases


def verify_skill_claims(claimed_skills: list[str], master_cv: MasterCV) -> list[str]:
    """FR-10a: Deterministic set check. Returns any unapproved skills.

    The allowed set expands grouped parentheticals so an older/manual YAML that
    still stores 'Python (Pandas, NumPy, Scikit-learn)' as one entry validates
    each atomic skill ('python', 'pandas', 'numpy', 'scikit-learn') correctly.
    """
    valid_names: set[str] = set()
    for s in master_cv.skills:
        valid_names.update(_expand_skill_aliases(s.en))
        valid_names.update(_expand_skill_aliases(s.fr))
    return [s for s in claimed_skills if s.lower() not in valid_names]


# Strict single-letter skill names (case-sensitive whole-word matching only).
_SINGLE_LETTER_SKILLS = {"R"}


def _extract_claimed_skills(text: str) -> list[str]:
    """Detects technical-skill mentions in prose using COMMON_TECH_SKILLS.

    Most skills use case-insensitive word-boundary matching so 'Go' won't match
    inside 'Google'. Single-letter skills like 'R' use strict whole-word AND
    case-sensitive matching (\\bR\\b uppercase only), so French prose and ordinary
    words never trigger a false positive.
    """
    if not text:
        return []
    claimed: list[str] = []
    for skill in COMMON_TECH_SKILLS:
        if skill in _SINGLE_LETTER_SKILLS:
            if re.search(
                r"(?<![A-Za-z0-9])" + re.escape(skill) + r"(?![A-Za-z0-9])", text
            ):
                claimed.append(skill)
            continue
        # \b anchors; re.escape guards special chars like 'C++', 'C#', '.NET'.
        pattern = r"(?<![A-Za-z0-9])" + re.escape(skill) + r"(?![A-Za-z0-9])"
        if re.search(pattern, text, flags=re.IGNORECASE):
            claimed.append(skill)
    return claimed


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


_NUMERAL_RE = re.compile(r"\d+(?:,\d+)*(?:\.\d+)?")


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
    return [
        num for num in found_numbers if not (_extract_numerals(num) & valid_numbers)
    ]


def select_cv_content(
    job_text: str,
    master_cv: MasterCV,
    api_key: str,
    language: Literal["en", "fr"] = "en",
    model: str = "openai/gpt-oss-120b",
) -> CVSelection:
    """
    Selects and ranks existing CV items by ID to match the target job posting (FR-8).
    Enforces deterministic validation so any hallucinated IDs are dropped.
    """
    schema = CVSelection.model_json_schema()

    catalog = {
        "skills": [
            {"id": s.id, "name": getattr(s, language)} for s in master_cv.skills
        ],
        "experience": [
            {
                "id": exp.id,
                "role": getattr(exp.role, language),
                "company": exp.company,
                "bullets": [
                    {"id": b.id, "text": getattr(b, language)} for b in exp.bullets
                ],
            }
            for exp in master_cv.experience
        ],
        "projects": [
            {
                "id": proj.id,
                "name": proj.name,
                "description": getattr(proj.description, language),
                "bullets": [
                    {"id": b.id, "text": getattr(b, language)} for b in proj.bullets
                ],
            }
            for proj in master_cv.projects
        ],
        "education": [
            {
                "id": edu.id,
                "degree": getattr(edu.degree, language),
                "institution": edu.institution,
            }
            for edu in master_cv.education
        ],
        "summaries": [
            {"id": s.id, "text": getattr(s, language)} for s in master_cv.summaries
        ],
    }

    has_summaries = bool(master_cv.summaries)

    summary_rule = (
        "6. summary_id: choose the single MOST relevant summary variant id from 'summaries' "
        "for this job. Use null only if 'summaries' is empty. NEVER write a new summary."
        if has_summaries
        else "6. summary_id: there are no summary variants; return null."
    )

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
    2. RANK, do not filter: Return ALL valid experience IDs in 'experiences' and, for EACH experience,
       ALL of its bullet IDs (every bullet in the catalog), ordered from MOST relevant to LEAST relevant.
       Do NOT drop bullets; the downstream layout engine decides what to trim.
    3. Return ALL valid project IDs in 'projects' and, for EACH project, ALL of its bullet IDs,
       ordered from MOST relevant to LEAST relevant. Projects are ordered from most to least relevant.
    4. Rank skill_ids from MOST to LEAST relevant for the target role.
    5. target_role must be a concise title adapted to the job description (e.g. 'Senior Backend Engineer').
    {summary_rule}
    """

    data = _call_llm_json(
        messages=[
            {
                "role": "system",
                "content": f"You are a strict data selector. Output JSON matching schema: {json.dumps(schema)}",
            },
            {"role": "user", "content": prompt},
        ],
        schema=schema,
        api_key=api_key,
        model=model,
        max_tokens=2600,
    )
    raw_selection = CVSelection(**data)

    valid_skill_ids = {s.id for s in master_cv.skills}
    valid_exp_order = [exp.id for exp in master_cv.experience]
    valid_exp_ids = {
        exp.id: [b.id for b in exp.bullets] for exp in master_cv.experience
    }
    valid_proj_order = [proj.id for proj in master_cv.projects]
    valid_proj_ids = {
        proj.id: [b.id for b in proj.bullets] for proj in master_cv.projects
    }
    valid_edu_ids = {edu.id for edu in master_cv.education}

    # --- Skills: keep ranked order, drop hallucinations, append any omitted. ---
    cleaned_skills: list[str] = []
    seen_skills: set[str] = set()
    for sid in raw_selection.skill_ids:
        if sid in valid_skill_ids and sid not in seen_skills:
            cleaned_skills.append(sid)
            seen_skills.add(sid)
    for s in master_cv.skills:
        if s.id not in seen_skills:
            cleaned_skills.append(s.id)
            seen_skills.add(s.id)

    def _scrub_experience_like(raw_blocks, valid_order, valid_bullets_map):
        """Keeps LLM ranking, drops hallucinated IDs, appends omitted valid items.

        For every experience/project that exists in the master CV we ALWAYS return
        a block (never dropping it), with all bullets ranked-first then any omitted
        valid bullets appended. Nothing is lost before the layout engine runs.
        """
        ordered_blocks = []
        seen_block_ids: set[str] = set()

        # First pass: honour the LLM's block order, validating IDs.
        for raw in raw_blocks:
            block_id = (
                raw.experience_id if hasattr(raw, "experience_id") else raw.project_id
            )
            if block_id in valid_bullets_map and block_id not in seen_block_ids:
                raw_bullet_ids = raw.bullet_ids
                allowed = valid_bullets_map[block_id]
                kept: list[str] = []
                seen_b: set[str] = set()
                for bid in raw_bullet_ids:
                    if bid in allowed and bid not in seen_b:
                        kept.append(bid)
                        seen_b.add(bid)
                # Append any valid bullets the LLM omitted, preserving CV order.
                for bid in allowed:
                    if bid not in seen_b:
                        kept.append(bid)
                        seen_b.add(bid)
                ordered_blocks.append((block_id, kept))
                seen_block_ids.add(block_id)

        # Second pass: append any valid blocks the LLM omitted entirely.
        for block_id in valid_order:
            if block_id not in seen_block_ids:
                ordered_blocks.append((block_id, list(valid_bullets_map[block_id])))
                seen_block_ids.add(block_id)

        return ordered_blocks

    ordered_experiences = _scrub_experience_like(
        raw_selection.experiences, valid_exp_order, valid_exp_ids
    )
    ordered_projects = _scrub_experience_like(
        raw_selection.projects, valid_proj_order, valid_proj_ids
    )

    cleaned_experiences = [
        SelectedExperience(experience_id=eid, bullet_ids=bids)
        for eid, bids in ordered_experiences
    ]
    cleaned_projects = [
        SelectedProject(project_id=pid, bullet_ids=bids)
        for pid, bids in ordered_projects
    ]

    cleaned_education = [
        eid for eid in raw_selection.education_ids if eid in valid_edu_ids
    ]

    # Education is a CORE section: it must never be empty when the Master CV has
    # degrees. Only fall back to ALL entries if the LLM returned none at all;
    # otherwise honour the selected subset (the renderer still shows every entry).
    if not cleaned_education and master_cv.education:
        print(
            "[!] No education entries survived validation; falling back to all education."
        )
        cleaned_education = [edu.id for edu in master_cv.education]

    # Summary selection: deterministic. Only pick an id that exists in
    # master_cv.summaries; otherwise fall back to the first one. Never generate.
    cleaned_summary_id: str | None = None
    if master_cv.summaries:
        valid_summary_ids = {s.id for s in master_cv.summaries}
        if raw_selection.summary_id in valid_summary_ids:
            cleaned_summary_id = raw_selection.summary_id
        else:
            cleaned_summary_id = master_cv.summaries[0].id

    return CVSelection(
        language=language,
        target_role=raw_selection.target_role,
        skill_ids=cleaned_skills,
        experiences=cleaned_experiences,
        projects=cleaned_projects,
        education_ids=cleaned_education,
        summary_id=cleaned_summary_id,
    )


# Leading greeting phrases the LLM may (incorrectly) emit as the first sentence
# even though the LaTeX template already prints the salutation.
_GREETING_PREFIX_RE = re.compile(
    r"^\s*(madame\s*,\s*monsieur|monsieur\s*,\s*madame|madame\s*,\s*monsieur\s*,|"
    r"mesdames\s*,\s*messieurs|madame|monsieur|mme|m\.|dr\.?|"
    r"dear\s+[^,\n:]+|to\s+whom\s+it\s+may\s+concern)\s*[,:]?\s*",
    re.IGNORECASE,
)

# Trailing sign-off phrases the LLM may emit as the final sentence even though
# the LaTeX template already prints the sign-off and the candidate's name.
_SIGNOFF_PREFIX_RE = re.compile(
    r"^\s*(cordialement|bien\s+cordialement|respectueusement|sinc[eè]rement|"
    r"sincerely|best\s+regards|kind\s+regards|regards|yours\s+(?:sincerely|faithfully)|"
    r"veuillez\s+agr[ée]er[^.\n]*)\s*[,:]?\s*",
    re.IGNORECASE,
)


def _strip_leading_greeting(text: str) -> str:
    """Removes a leading salutation line from cover-letter body text."""
    if not text:
        return text
    return _GREETING_PREFIX_RE.sub("", text, count=1).strip()


def _strip_trailing_signoff(text: str) -> str:
    """Removes a leading/embedded sign-off phrase from the first body sentence."""
    if not text:
        return text
    return _SIGNOFF_PREFIX_RE.sub("", text, count=1).strip()


def _normalize_cover_letter_body(letter: CoverLetterDraft) -> CoverLetterDraft:
    """Strips greeting prefixes and sign-off text from the LLM's sentence array.

    The LaTeX template renders the salutation and sign-off itself, so any
    greeting/sign-off the model placed inside 'sentences' would otherwise appear
    twice. This is deterministic and returns a new CoverLetterDraft.
    """
    cleaned_sentences = []
    for sentence in letter.sentences:
        text = sentence.text or ""
        text = _strip_leading_greeting(text)
        # Sign-off may be its own trailing sentence or prefixed to it.
        text = _strip_trailing_signoff(text)
        # Drop a sentence that became the candidate's bare name after stripping.
        if not text.strip():
            continue
        cleaned_sentences.append(
            CoverLetterSentence(
                paragraph=sentence.paragraph,
                text=text.strip(),
                based_on=sentence.based_on,
            )
        )

    return letter.model_copy(update={"sentences": cleaned_sentences})


def verify_cover_letter(letter: CoverLetterDraft, master_cv: MasterCV) -> list[str]:
    """Runs deterministic verification on citations, numerals, and skill claims."""
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
        errors.append(
            f"Fabricated numerals outside facts/allowlist: {', '.join(invalid_numerals)}"
        )

    # 3. Check skill claims (FR-10a): any technical skill named in the letter that
    #    is not present in the Master CV is treated as a fabricated claim.
    claimed_skills = _extract_claimed_skills(full_text)
    unapproved_skills = verify_skill_claims(claimed_skills, master_cv)
    if unapproved_skills:
        errors.append(
            f"Skills claimed but not present in Master CV: {', '.join(unapproved_skills)}"
        )

    return errors


def _build_cover_letter_catalog(
    master_cv: MasterCV,
    language: Literal["en", "fr"],
    selection: "CVSelection | None" = None,
    fit_result: "FitAnalysis | None" = None,
) -> dict:
    """Builds a compact, single-language, focused context payload for the LLM.

    Groq's free tier caps a single request at 8,000 TPM (prompt + max_tokens), so we:

    * resolve every localized field to the ACTIVE ``language`` only (never serialize
      the inactive one), and
    * when a ``selection`` is provided, pass a focused top-ranked slice rather than
      the entire CV (which now preserves all ranked skills/bullets):

        - top 12 ranked skills,
        - top 2 experiences with all their ranked bullets, plus the 3rd experience
          capped at its top 2 bullets,
        - top 2 projects with up to 2 ranked bullets each,
        - the selected summary (if any),
        - facts, plus ``fit_result`` strengths / covered_keywords when supplied.

    The full selection is still used for CV compilation elsewhere; only this
    cover-letter context is trimmed.
    """
    lang = language
    skills_by_id = {s.id: s for s in master_cv.skills}
    exp_by_id = {e.id: e for e in master_cv.experience}
    proj_by_id = {p.id: p for p in master_cv.projects}

    catalog: dict = {"candidate_name": master_cv.personal_info.name}

    # Facts (already language-neutral scalars).
    catalog["facts"] = master_cv.facts.model_dump()

    # --- Skills: top 12 ranked, resolved to the active language only. ---
    if selection is not None:
        ranked_skill_ids = selection.skill_ids
    else:
        ranked_skill_ids = [s.id for s in master_cv.skills]
    top_skill_ids = ranked_skill_ids[:12]
    skills_out = []
    for sid in top_skill_ids:
        skill = skills_by_id.get(sid)
        if skill is None:
            continue
        skills_out.append({"id": skill.id, "name": getattr(skill, lang)})
    catalog["skills"] = skills_out

    # --- Experience: top 2 fully, 3rd capped at 2 bullets. ---
    if selection is not None:
        sel_exps = selection.experiences
    else:
        sel_exps = [
            SelectedExperience(experience_id=e.id, bullet_ids=[b.id for b in e.bullets])
            for e in master_cv.experience
        ]
    exp_out = []
    for rank_idx, sel_exp in enumerate(sel_exps):
        exp = exp_by_id.get(sel_exp.experience_id)
        if exp is None:
            continue
        bullet_lookup = {b.id: b for b in exp.bullets}
        if rank_idx < 2:
            bullet_ids = sel_exp.bullet_ids
        elif rank_idx == 2:
            bullet_ids = sel_exp.bullet_ids[:2]
        else:
            continue
        bullets_out = [
            {"id": bid, "text": getattr(bullet_lookup[bid], lang)}
            for bid in bullet_ids
            if bid in bullet_lookup
        ]
        exp_out.append(
            {
                "id": exp.id,
                "role": getattr(exp.role, lang),
                "company": exp.company,
                "bullets": bullets_out,
            }
        )
    catalog["experience"] = exp_out

    # --- Projects: top 2, up to 2 ranked bullets each. ---
    if selection is not None:
        sel_projs = selection.projects
    else:
        sel_projs = [
            SelectedProject(project_id=p.id, bullet_ids=[b.id for b in p.bullets])
            for p in master_cv.projects
        ]
    proj_out = []
    for sel_proj in sel_projs[:2]:
        proj = proj_by_id.get(sel_proj.project_id)
        if proj is None:
            continue
        bullet_lookup = {b.id: b for b in proj.bullets}
        bullets_out = [
            {"id": bid, "text": getattr(bullet_lookup[bid], lang)}
            for bid in sel_proj.bullet_ids[:2]
            if bid in bullet_lookup
        ]
        proj_out.append({"id": proj.id, "name": proj.name, "bullets": bullets_out})
    catalog["projects"] = proj_out

    # --- Summary: the selected grounded variant for the active language. ---
    if master_cv.summaries:
        chosen = None
        if selection is not None and selection.summary_id:
            for s in master_cv.summaries:
                if s.id == selection.summary_id:
                    chosen = s
                    break
        if chosen is None:
            chosen = master_cv.summaries[0]
        summary_text = getattr(chosen, lang) or ""
        if summary_text.strip():
            catalog["summary"] = summary_text.strip()

    # --- Fit-analysis signals (job-tailoring hints, already language-neutral). ---
    if fit_result is not None:
        if fit_result.strengths:
            catalog["fit_strengths"] = list(fit_result.strengths)
        if fit_result.covered_keywords:
            catalog["fit_covered_keywords"] = list(fit_result.covered_keywords)

    return catalog


def generate_cover_letter(
    job_text: str,
    master_cv: MasterCV,
    api_key: str,
    language: Literal["en", "fr"] = "en",
    model: str = "openai/gpt-oss-120b",
    max_retries: int = 2,
    selection: "CVSelection | None" = None,
    fit_result: "FitAnalysis | None" = None,
) -> CoverLetter:
    """
    FR-9 & FR-10: Generates a tailored 3-4 paragraph cover letter where every factual claim
    cites a Master CV ID. Automatically triggers re-prompting if verification checks fail.

    To stay within Groq's 8,000 TPM single-request limit, the context payload is
    built by :func:`_build_cover_letter_catalog`: single-language only, a focused
    top-ranked slice of the CV (when ``selection`` is provided), and compact JSON.
    """
    schema = CoverLetterDraft.model_json_schema()

    catalog = _build_cover_letter_catalog(
        master_cv, language, selection=selection, fit_result=fit_result
    )
    # Compact JSON (no indentation/whitespace tokens) keeps the prompt small.
    catalog_json = json.dumps(catalog, separators=(",", ":"))

    base_prompt = f"""
You are an expert technical cover letter writer.
Draft a professional, compelling, 3-to-4 paragraph cover letter tailored to the job description.

TARGET LANGUAGE: {language}

CANDIDATE DATA & VALID IDs:
{catalog_json}

JOB DESCRIPTION:
\"\"\"{job_text}\"\"\"

CRITICAL HONESTY & FORMAT RULES:
1. Provide the letter as a flat array of 'sentences'. Tag each sentence with its 'paragraph' number (1, 2, 3, or 4).
2. Every sentence making a factual statement (experience, metrics, tooling) MUST populate 'based_on' with valid IDs from candidate data.
3. If a sentence is an opening greeting, transition, or closing signoff, 'based_on' must be an empty list [].
4. DO NOT invent metrics or numbers. Only use numbers present in the candidate facts or bullets.
5. DO NOT reference technologies or skills not listed in the candidate data.
6. NEVER name a programming language, framework, or library that is absent from the candidate's skills, experience, and projects — even if the job description mentions it. This includes single-letter languages like 'R'. Derive the language from the candidate data only.
7. The 'sentences' array must contain BODY PARAGRAPHS ONLY. Do NOT include a greeting/salutation (e.g. 'Madame, Monsieur,' / 'Dear Hiring Team,') and do NOT include a closing sign-off (e.g. 'Cordialement,' / 'Sincerely,'). The greeting and sign-off are rendered separately by the document template.
"""

    messages = [
        {
            "role": "system",
            "content": f"You are a strict technical writer with zero-hallucination constraints. Output JSON matching schema: {json.dumps(schema)}",
        },
        {"role": "user", "content": base_prompt},
    ]

    for attempt in range(max_retries + 1):
        raw_content = None
        data = _call_llm_json(
            messages=messages,
            schema=schema,
            api_key=api_key,
            model=model,
            max_tokens=2600,
        )
        # Keep a serialized copy so we can echo it back as the assistant turn on retries.
        raw_content = json.dumps(data)

        draft = CoverLetterDraft(**data)

        # Strip any greeting/sign-off the model echoed into the body so the
        # template's own salutation and sign-off are not duplicated.
        draft = _normalize_cover_letter_body(draft)

        errors = verify_cover_letter(draft, master_cv)

        if not errors:
            return CoverLetter(
                **draft.model_dump(), verification_passed=True, verification_warnings=[]
            )

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

    return CoverLetter(
        **draft.model_dump(), verification_passed=False, verification_warnings=errors
    )


# Legitimate hyphenated terms that must NOT be melded when split across a line
# break. Compared case-insensitively against the joined "left-right" form.
PRESERVE_HYPHEN_TERMS = {
    "scikit-learn",
    "text-to-sql",
    "cross-functional",
    "cross-team",
    "cross-platform",
    "end-to-end",
    "front-end",
    "back-end",
    "full-stack",
    "real-time",
    "high-throughput",
    "state-machine",
    "test-driven",
    "event-driven",
    "data-driven",
    "self-service",
    "open-source",
    "e-commerce",
    "co-operative",
    "go-to-market",
    "b2b",
    "b2c",
    "ci-cd",
    "dev-ops",
    "devops",
    "ai-ml",
    "fine-tuning",
    "pre-processing",
    "micro-services",
    "microservices",
    "well-known",
    "long-term",
    "short-term",
}

_SOFT_HYPHEN_RE = re.compile(r"(\w+)-[ \t]*\n[ \t]*(\w+)")

# Literal currency/backslash artifacts left by PDF extraction. A source token
# like "\$0" carries a stray backslash; without normalization the "$" still gets
# escaped to "\$" while the original "\" remains, producing a broken "0\$"
# (or "\textbackslash{}$") artifact in the compiled output. Collapsing "\$" -> "$"
# first lets LaTeX cleanly render a single "$".
_BACKSLASH_DOLLAR_RE = re.compile(r"\\\$")


def _normalize_pdf_artifacts(text: str) -> str:
    """Normalizes literal ``\\$`` sequences (common in currency amounts) to ``$``.

    Prevents the doubled-escape artifact (``\\textbackslash{}$`` / ``0\\$``) in
    rendered output by removing the stray backslash while preserving the dollar
    symbol so LaTeX escaping can cleanly emit ``$``.
    """
    if not text:
        return text
    return _BACKSLASH_DOLLAR_RE.sub("$", text)


def _dehyphenate(text: str) -> str:
    """Joins soft line-break hyphens ('Chro-\\nmaDB' -> 'ChromaDB').

    A lowercase->lowercase join across a line break is melded UNLESS the joined
    term is a known legitimate compound (e.g. 'Scikit-learn', 'cross-functional',
    'Text-to-SQL'), in which case the hyphen is preserved. Capitalised joins
    (acronyms / compound words) always keep the hyphen.
    """

    def _join(match: re.Match) -> str:
        left, right = match.group(1), match.group(2)
        candidate = f"{left}-{right}".lower()
        if candidate in PRESERVE_HYPHEN_TERMS:
            return f"{left}-{right}"
        # Preserve hyphen when either side is capitalised (acronym / proper noun).
        if left[-1].isupper() or right[0].isupper():
            return f"{left}-{right}"
        return f"{left}{right}"

    return _SOFT_HYPHEN_RE.sub(_join, text)


def _page_text_in_reading_order(page: "pymupdf.Page") -> str:
    """Extracts page text sorted by visual position, with spacing between spans.

    Blocks are sorted by (rounded top, left) so right-aligned content (dates on
    education entries) stays attached to its own line instead of sinking to the
    bottom. Horizontally separated spans on the same line are joined with a space
    so adjacent dates don't collide (e.g. '11/2025 11/2026' rather than
    '11/202511/2026').
    """
    lines_out: list[str] = []
    # Use 'dict' for span-level geometry, then reflow by visual order.
    block_data = page.get_text("dict")
    line_records = []
    for block in block_data.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text_parts = [s.get("text", "") for s in spans]
            bbox = line.get("bbox", (0, 0, 0, 0))
            line_text = ""
            prev_x1 = None
            for idx, span in enumerate(spans):
                s_text = span.get("text", "")
                s_bbox = span.get("bbox", (0, 0, 0, 0))
                if idx > 0:
                    # Insert a space when spans are horizontally separated.
                    if prev_x1 is not None and s_bbox[0] - prev_x1 > 1.0:
                        line_text += " "
                line_text += s_text
                prev_x1 = s_bbox[2]
            if line_text.strip():
                line_records.append(
                    (round(bbox[1], 1), round(bbox[0], 1), line_text.rstrip())
                )

    line_records.sort(key=lambda r: (r[0], r[1]))
    for _, _, text in line_records:
        lines_out.append(text)
    return "\n".join(lines_out)


def extract_text_from_pdf(pdf_source: bytes | str) -> str:
    """Extracts raw text from a PDF path or raw bytes, preserving reading order.

    Sorts lines by visual position so right-aligned dates stay with their entry,
    then repairs soft-hyphen line wraps.
    """
    if isinstance(pdf_source, (bytes, bytearray)):
        doc = pymupdf.open(stream=pdf_source, filetype="pdf")
    else:
        doc = pymupdf.open(pdf_source)

    try:
        pages_text = [_page_text_in_reading_order(page) for page in doc]
    finally:
        doc.close()

    extracted_text = "\n".join(pages_text).strip()
    return _normalize_pdf_artifacts(_dehyphenate(extracted_text))


def _anchor_text_for_rect(page: "pymupdf.Page", rect: "pymupdf.Rect") -> str:
    """Returns the page text whose span geometry overlaps/nears the link rectangle."""
    words = page.get_text("words")  # (x0, y0, x1, y1, word, block, line, word_no)
    # Expand the rect slightly to catch anchor text that renders just above/around it.
    target = pymupdf.Rect(rect.x0 - 2, rect.y0 - 2, rect.x1 + 2, rect.y1 + 2)
    hits = []
    for w in words:
        w_rect = pymupdf.Rect(w[0], w[1], w[2], w[3])
        if w_rect.intersects(target):
            hits.append((round(w[1], 1), round(w[0], 1), w[4]))
    hits.sort(key=lambda t: (t[0], t[1]))
    return " ".join(h[2] for h in hits).strip()


def extract_pdf_content(pdf_source: bytes | str) -> tuple[str, list[PdfLink]]:
    """Extracts reading-ordered text and anchored hyperlinks from a PDF.

    Returns ``(text, links)`` where each link carries the visible anchor text found
        near its rectangle (e.g. the 'Live demo' label under a project), so downstream
    extraction can map URLs to the correct schema field.
    """
    if isinstance(pdf_source, (bytes, bytearray)):
        doc = pymupdf.open(stream=pdf_source, filetype="pdf")
    else:
        doc = pymupdf.open(pdf_source)

    links: list[PdfLink] = []
    seen: set[str] = set()
    try:
        pages_text = []
        for page in doc:
            pages_text.append(_page_text_in_reading_order(page))
            for link in page.get_links():
                uri = link.get("uri")
                if not uri or uri in seen:
                    continue
                anchor = ""
                if link.get("from") is not None:
                    anchor = _anchor_text_for_rect(page, link["from"])
                seen.add(uri)
                links.append(PdfLink(anchor_text=anchor, url=uri))
    finally:
        doc.close()

    text = _normalize_pdf_artifacts(_dehyphenate("\n".join(pages_text).strip()))
    return text, links


_EDUCATION_KEYWORD_RE = re.compile(
    r"\b(University|Université|College|Institut|Institute|School|Master|Master's|"
    r"Bachelor|Bachelor's|Degree|BSc|B\.Sc|MSc|M\.Sc|PhD|Ph\.D|Licence|Doctorate|Diploma)\b",
    re.IGNORECASE,
)


def _normalize_for_grounding(text: str) -> str:
    """Lowercases and collapses whitespace for tolerant substring grounding checks."""
    return re.sub(r"\s+", " ", (text or "").lower())


def ground_extraction(cv: MasterCV, source_text: str) -> list[str]:
    """Deterministic post-extraction grounding check (Issue 5).

    Returns a list of human-readable warnings when extracted content looks
    ungrounded against the normalized source PDF text:
      - a bullet claims numerals that do not appear anywhere in the source;
      - education is empty despite education keywords being present in the source.
    """
    warnings: list[str] = []
    source_norm = _normalize_for_grounding(source_text)
    source_numerals: set[str] = set()
    for token in _NUMERAL_RE.findall(source_text or ""):
        source_numerals.update(_extract_numerals(token))

    # 1. Numeric claims in bullets must be traceable to the source.
    for exp in cv.experience:
        for b in exp.bullets:
            for key in ("en", "fr"):
                num_text = getattr(b, key, "") or ""
                for num in _NUMERAL_RE.findall(num_text):
                    if not (_extract_numerals(num) & source_numerals):
                        warnings.append(
                            f"Bullet '{b.id}' ({key}) cites number '{num}' not found in the source PDF."
                        )
    for proj in cv.projects:
        for b in proj.bullets:
            for key in ("en", "fr"):
                num_text = getattr(b, key, "") or ""
                for num in _NUMERAL_RE.findall(num_text):
                    if not (_extract_numerals(num) & source_numerals):
                        warnings.append(
                            f"Project bullet '{b.id}' ({key}) cites number '{num}' not found in the source PDF."
                        )

    # 2. Education should not be empty if the PDF clearly mentions degrees.
    if not cv.education and _EDUCATION_KEYWORD_RE.search(source_text or ""):
        warnings.append(
            "Education section is empty but the PDF text mentions degrees/universities. "
            "Re-check the education extraction."
        )

    return warnings


def parse_pdf_to_master_cv(
    pdf_text: str,
    api_key: str,
    model: str = "openai/gpt-oss-120b",
    hyperlinks: list[PdfLink] | None = None,
) -> tuple[MasterCV, list[str]]:
    """Parses unstructured CV text into a validated MasterCV plus grounding warnings.

    Generates unique ID slugs for every item and provides bilingual EN/FR fields.
    Returns ``(master_cv, warnings)`` where warnings come from the deterministic
    grounding check in :func:`ground_extraction`.
    """
    schema = MasterCV.model_json_schema()

    link_block = ""
    if hyperlinks:
        pairs = "\n".join(
            f'  - anchor: "{l.anchor_text}" -> url: "{l.url}"' for l in hyperlinks
        )
        link_block = (
            "\n\nEMBEDDED HYPERLINKS (anchor text -> URL), extracted from the PDF:\n"
            f"{pairs}\n"
            "Map these to the correct fields: project URLs to that project's 'url', "
            "and profile URLs (LinkedIn/GitHub/portfolio) to personal_info."
        )

    prompt = f"""
You are an expert ATS data engineer. Convert this unstructured resume into a Master CV definition.
Output ONLY content that is explicitly present in the resume text below.

RESUME TEXT:
\"\"\"{pdf_text}\"\"\"{link_block}

STRICT EXTRACTION RULES (zero fabrication — this is the core guarantee of the product):
1. BULLETS: Extract ONLY bullet points that explicitly exist in the source text for that role
   or project. Preserve the EXACT bullet count. NEVER invent, split, merge, paraphrase-expand,
   or pad bullets to match other roles. If a role has 2 bullets, output exactly 2.
2. NO DUPLICATION: A project's 'description' must be a SEPARATE non-bullet summary line if one
   exists. Do NOT copy a bullet into 'description' and then also emit a paraphrased bullet.
   If there is no separate summary line, set 'description' to the verbatim first line WITHOUT
   also repeating it in 'bullets' (omit it from bullets).
3. LOCATIONS: Only populate a 'location' (personal_info or experience) if a location is
   EXPLICITLY written in that header or entry. Otherwise use null. Never infer a city from an
   employer name or copy a job location into personal_info.
4. PERSONAL INFO: name, email, phone, linkedin, github, portfolio_url. Only fill fields that
   are actually present; use null otherwise.
5. SKILLS: Each distinct technical tool/language/framework once, id 'sk_<slug>'.
   DECOMPOSE grouped skills into ATOMIC entries: 'Python (Pandas, NumPy, Scikit-learn)'
   becomes FOUR skills: Python (sk_python), Pandas (sk_pandas), NumPy (sk_numpy),
   Scikit-learn (sk_scikit_learn). Never keep parenthetical groups as a single skill.
   Assign each atomic skill a bilingual 'category' (both en and fr), choosing from:
     - 'Languages & Databases' / 'Langages & Bases de données'
     - 'Python & ML' / 'Python & ML'
     - 'BI & Analytics' / 'BI & Analytique'
     - 'Tools' / 'Outils'
     - 'Spoken Languages' / 'Langues'
6. SPOKEN LANGUAGES: Extract spoken languages and proficiency levels stated ANYWHERE in
   the CV (Skills, Summary, header) as atomic 'skills' entries preserving the level tag,
   e.g. 'Arabic (native)'/'Arabe (natif)', 'English (C1)'/'Anglais (C1)',
   'French (B2)'/'Français (B2)'. Do NOT drop languages just because they are in a Summary.
   These should carry category 'Spoken Languages' / 'Langues'.
7. EXPERIENCE: EVERY work entry. Translate each bullet faithfully to French ('fr'); the French
   MUST preserve all numbers/metrics exactly.
8. PROJECTS: Extract EVERY project present in the document (do not stop early). One entry per
   project with id 'proj_<slug>'; attach the matching hyperlink as 'url' when available.
9. EDUCATION: Extract EVERY degree/qualification listed (Bachelor's, Master's, PhD, diplomas).
   ONE entry per degree, id 'edu_<slug>', with institution, full degree title, graduation year.
   Right-aligned dates belong to that degree's line — do not drop the education section.
10. If 'experience' or 'education' is mentioned anywhere, it MUST NOT be empty.
11. SUMMARIES: If the CV opens with a professional summary/profile/objective paragraph,
    extract it into 'summaries' as id 'sum_default' with faithful 'en' and 'fr' translations.
    Strip any trailing 'Languages: ...' / 'Langues : ...' sentence from the summary text
    (those languages are captured separately as skills). If there is no summary paragraph,
    return an empty 'summaries' list [].
"""

    data = _call_llm_json(
        messages=[
            {
                "role": "system",
                "content": f"You are a strict data transformation engine with zero-hallucination constraints. Output JSON matching schema: {json.dumps(schema)}",
            },
            {"role": "user", "content": prompt},
        ],
        schema=schema,
        api_key=api_key,
        model=model,
        temperature=0.0,
        max_tokens=6000,
    )
    cv = MasterCV(**data)
    warnings = ground_extraction(cv, pdf_text)
    return cv, warnings


def _normalize_extracted_skills(cv: MasterCV) -> MasterCV:
    """Deterministically splits any grouped skills into atomic Skill entries.

    Handles the case where the LLM still returns a grouped string like
    'Python (Pandas, NumPy, Scikit-learn)' (or a slash/comma list). Atomic skills
    get clean ids ('sk_python', 'sk_pandas', ...) and deduplicated EN/FR labels.
    Proficiency parentheticals (e.g. 'English (C1)', 'Arabic (native)') are
    preserved verbatim. Each atomic skill inherits its source skill's ``category``
    (grouped skills share the parent category). Returns a new MasterCV; the input
    is not mutated.
    """
    seen_en: set[str] = set()
    seen_ids: set[str] = set()
    new_skills: list[Skill] = []

    for skill in cv.skills:
        en_atoms = _split_skill_string(skill.en)
        fr_atoms = _split_skill_string(skill.fr)

        # Pair up EN/FR atoms positionally; fall back to the raw label when the
        # counts don't line up so we never lose information.
        if len(en_atoms) == len(fr_atoms):
            pairs = list(zip(en_atoms, fr_atoms))
        else:
            pairs = [(a, a) for a in en_atoms] or [(skill.en, skill.fr)]

        for en_name, fr_name in pairs:
            key = en_name.strip().lower()
            if not key or key in seen_en:
                continue
            seen_en.add(key)

            new_id = _skill_slug(en_name)
            if new_id in seen_ids:
                suffix = 2
                while f"{new_id}_{suffix}" in seen_ids:
                    suffix += 1
                new_id = f"{new_id}_{suffix}"
            seen_ids.add(new_id)

            new_skills.append(
                Skill(
                    id=new_id,
                    en=en_name.strip(),
                    fr=fr_name.strip(),
                    category=skill.category,
                )
            )

    return cv.model_copy(update={"skills": new_skills})


# Trailing "Languages: ..." / "Langues : ..." segment to strip from a summary when
# those spoken languages are already captured as skills.
_SUMMARY_LANGUAGES_RE = re.compile(
    r"(?:^|[\s.])(?:languages?|langues?)\s*[:\-–]\s*[^.]*\.?\s*$",
    re.IGNORECASE,
)


def _strip_languages_from_summary(text: str) -> str:
    """Removes a trailing 'Languages: ...' clause from a summary paragraph."""
    if not text:
        return text
    stripped = _SUMMARY_LANGUAGES_RE.sub("", text).strip()
    return stripped or text.strip()


def _normalize_summaries(cv: MasterCV) -> MasterCV:
    """Cleans extracted summary variants deterministically.

    - Assigns the canonical id ``sum_default`` to a single summary (or stable ids
      to multiple), so the selector/renderer can address them.
    - Strips a trailing 'Languages: ...' sentence from EN and FR text since those
      languages are captured separately as skills.
    - Drops entirely empty summaries. Returns a new MasterCV; input not mutated.
    """
    if not cv.summaries:
        return cv

    cleaned = []
    seen_ids: set[str] = set()
    for idx, summary in enumerate(cv.summaries):
        en = _strip_languages_from_summary(summary.en or "")
        fr = _strip_languages_from_summary(summary.fr or "")
        if not en.strip() and not fr.strip():
            continue
        sid = (summary.id or "").strip() or f"sum_{idx}"
        if sid in seen_ids:
            suffix = 2
            while f"{sid}_{suffix}" in seen_ids:
                suffix += 1
            sid = f"{sid}_{suffix}"
        seen_ids.add(sid)
        cleaned.append(SummaryVariant(id=sid, en=en, fr=fr))

    # Canonical id for the (single) primary summary, preserving extraction order.
    if cleaned and cleaned[0].id != "sum_default":
        cleaned[0] = cleaned[0].model_copy(update={"id": "sum_default"})

    return cv.model_copy(update={"summaries": cleaned})


def build_master_cv_from_pdf(
    pdf_source: bytes | str, api_key: str, model: str = "openai/gpt-oss-120b"
) -> tuple[MasterCV, list[str]]:
    """Shared PDF -> MasterCV pipeline used by both the CLI and the Streamlit UI.

    Extracts reading-ordered text plus anchored hyperlinks, runs LLM extraction,
    deterministically splits any grouped skills into atomic entries (preserving
    each skill's bilingual category), normalizes summary variants, and returns
    ``(master_cv, warnings)`` where warnings are grounding issues.
    """
    text, links = extract_pdf_content(pdf_source)
    cv, warnings = parse_pdf_to_master_cv(
        text, api_key=api_key, model=model, hyperlinks=links
    )
    cv = _normalize_extracted_skills(cv)
    cv = _normalize_summaries(cv)
    return cv, warnings


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
