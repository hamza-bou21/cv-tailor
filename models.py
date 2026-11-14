from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# ---------------------------------------------------------
# 1. Master CV Models (Parsing the YAML)
# ---------------------------------------------------------


class BilingualText(BaseModel):
    en: str
    fr: str


# Alias kept for callers that refer to a generic localized text field, e.g. a
# skill's optional category. Identical shape to BilingualText, so existing YAML
# and code keep working unchanged.
LocalizedText = BilingualText


class PersonalInfo(BaseModel):
    name: str
    email: str
    linkedin: str
    github: str
    phone: Optional[str] = None
    location: Optional[str] = None
    portfolio_url: Optional[str] = None


class Facts(BaseModel):
    years_experience_total: int
    years_experience_backend: Optional[int] = None
    max_team_size_led: Optional[int] = None
    highest_traffic_handled_rps: Optional[int] = None
    revenue_impact_usd: Optional[int] = None


class Skill(BaseModel):
    id: str
    en: str
    fr: str
    # Optional bilingual grouping category (e.g. Languages & Databases). Defaults
    # to None so older YAML profiles without a category still validate cleanly.
    category: Optional[LocalizedText] = None


class Bullet(BaseModel):
    id: str
    en: str
    fr: str


class Experience(BaseModel):
    id: str
    company: str
    location: Optional[str] = None
    role: BilingualText
    start_date: str
    end_date: str
    bullets: List[Bullet] = Field(default_factory=list)


class Project(BaseModel):
    id: str
    name: str
    url: Optional[str] = None
    description: BilingualText
    bullets: List[Bullet] = Field(default_factory=list)


class Education(BaseModel):
    id: str
    institution: str
    degree: BilingualText
    graduation_year: str


class SummaryVariant(BaseModel):
    """A grounded professional summary/profile paragraph extracted from the source CV.

    Stored verbatim-derived (with an FR translation) so the renderer never invents
    positioning text; the selector only *chooses* among these, never writes new ones.
    """

    id: str
    en: str
    fr: str


# ---------------------------------------------------------
# 1b. PDF extraction helpers
# ---------------------------------------------------------


class PdfLink(BaseModel):
    """A hyperlink found in a PDF, anchored to the text it visually overlays."""

    anchor_text: str = Field(
        default="",
        description="Visible text near/under the link rectangle (e.g. 'Live demo').",
    )
    url: str


# ---------------------------------------------------------
# 2. LLM Contracts (The exact JSON we force the AI to return)
# ---------------------------------------------------------


class FitAnalysis(BaseModel):
    """Output contract for the initial job analysis pass."""

    fit_score: int = Field(..., ge=0, le=100)
    verdict: Literal["apply", "stretch", "skip"]
    strengths: List[str]
    gaps: List[str]
    covered_keywords: List[str]
    missing_keywords: List[str]


class CoverLetterClaim(BaseModel):
    """Output contract for the verification pass (cite-then-validate)."""

    sentence: str
    based_on_ids: List[str] = Field(
        description="IDs from the Master CV that support this sentence."
    )
    is_supported: bool = Field(
        description="True if the cited bullets completely support the claim."
    )
    explanation: str = Field(
        description="Brief explanation of why it is or isn't supported."
    )


class VerificationReport(BaseModel):
    """The final verification result of the generated cover letter."""

    claims: List[CoverLetterClaim]
    unsupported_claims_found: bool


class MasterCV(BaseModel):
    personal_info: PersonalInfo
    facts: Facts
    skills: List[Skill] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    # Optional grounded summary variants. Defaults to empty for backward compat.
    summaries: List[SummaryVariant] = Field(default_factory=list)


class DocumentRecord(BaseModel):
    """Schema tracking generated documents and provenance audit trails."""

    id: Optional[str] = None
    job_id: str
    doc_type: Literal["cv", "cover_letter", "letter"]
    language: Literal["en", "fr"]
    pdf_path: str
    tex_path: str
    date_generated: str
    verification_warnings: List[str] = Field(default_factory=list)
    manually_edited: bool = Field(
        default=False,
        description="Provenance marker: True if manual edits occurred post-generation",
    )
    edited_fields: Optional[dict] = Field(
        default=None, description="Tracks modified fields when manually_edited is True"
    )


class SelectedExperience(BaseModel):
    experience_id: str
    bullet_ids: List[str]


class SelectedProject(BaseModel):
    project_id: str
    bullet_ids: List[str]


class CVSelection(BaseModel):
    """Output contract for LLM bullet and skill selection (FR-8)."""

    language: Literal["en", "fr"]
    target_role: str
    skill_ids: List[str] = Field(description="Selected skill IDs ordered by relevance.")
    experiences: List[SelectedExperience] = Field(
        description="Selected experience blocks and their chosen bullet IDs."
    )
    projects: List[SelectedProject] = Field(
        default_factory=list,
        description="Selected project IDs and their chosen bullet IDs.",
    )
    education_ids: List[str] = Field(
        default_factory=list, description="Selected education IDs."
    )
    summary_id: Optional[str] = Field(
        default=None,
        description="ID of the chosen grounded summary variant (or None if no summaries exist).",
    )


class CoverLetterSentence(BaseModel):
    paragraph: int = Field(default=1, description="Paragraph index: 1, 2, 3, or 4.")
    text: str = Field(description="Sentence prose.")
    based_on: List[str] = Field(
        default_factory=list,
        description="IDs of skills, experience bullets, or projects justifying this sentence. Empty if purely conversational/greeting.",
    )


class CoverLetterDraft(BaseModel):
    """Schema provided to the LLM (flat sentences array, zero nested paragraph objects)."""

    language: Literal["en", "fr"]
    company: str
    target_role: str
    salutation: str
    sentences: List[CoverLetterSentence] = Field(
        description="All sentences of the letter in order, each tagged with its paragraph number (1 to 4)."
    )
    signoff: str


class CoverLetter(CoverLetterDraft):
    """Full application model with deterministic verification status."""

    verification_passed: bool = False
    verification_warnings: List[str] = Field(default_factory=list)
