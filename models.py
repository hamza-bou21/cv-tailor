from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# ---------------------------------------------------------
# 1. Master CV Models (Parsing the YAML)
# ---------------------------------------------------------

class BilingualText(BaseModel):
    en: str
    fr: str

class PersonalInfo(BaseModel):
    name: str
    email: str
    linkedin: str
    github: str
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

class Bullet(BaseModel):
    id: str
    en: str
    fr: str

class Experience(BaseModel):
    id: str
    company: str
    location: str
    role: BilingualText
    start_date: str
    end_date: str
    bullets: List[Bullet] = Field(default_factory=list) # <--- Update this line

class Project(BaseModel):
    id: str
    name: str
    url: Optional[str] = None
    description: BilingualText
    bullets: List[Bullet] = Field(default_factory=list) # <--- Update this line

class Education(BaseModel):
    id: str
    institution: str
    degree: BilingualText
    graduation_year: str

class MasterCV(BaseModel):
    personal_info: PersonalInfo
    facts: Facts 
    skills: List[Skill]
    experience: List[Experience]
    projects: List[Project]
    education: List[Education]


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
    based_on_ids: List[str] = Field(description="IDs from the Master CV that support this sentence.")
    is_supported: bool = Field(description="True if the cited bullets completely support the claim.")
    explanation: str = Field(description="Brief explanation of why it is or isn't supported.")

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