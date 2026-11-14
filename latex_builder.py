import os
import re
import shutil
import subprocess
import datetime
from collections import defaultdict
from typing import Optional

import pymupdf
from models import MasterCV, CVSelection, CoverLetter

LATEX_ESCAPE_MAP = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

# French month names for localizing the cover-letter date.
_FRENCH_MONTHS = [
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
]

# Greeting prefixes that may leak into the first body sentence. The template
# already prints the salutation, so these are stripped defensively.
_GREETING_PREFIX_RE = re.compile(
    r"^\s*(madame\s*,\s*monsieur|monsieur\s*,\s*madame|mesdames\s*,\s*messieurs|"
    r"madame|monsieur|dear\s+[^,\n:]+|to\s+whom\s+it\s+may\s+concern)\s*[,:]?\s*",
    re.IGNORECASE,
)

# Sign-off phrases that may leak into the body or replace the template sign-off.
_SIGNOFF_PREFIX_RE = re.compile(
    r"^\s*(cordialement|bien\s+cordialement|respectueusement|sinc[eè]rement|"
    r"sincerely|best\s+regards|kind\s+regards|regards|yours\s+(?:sincerely|faithfully))\s*[,:]?\s*",
    re.IGNORECASE,
)


def _strip_leading_greeting(text: str) -> str:
    """Removes a leading salutation from body text."""
    if not text:
        return text
    return _GREETING_PREFIX_RE.sub("", text, count=1).strip()


def _localized_date(lang: str, today: Optional[datetime.date] = None) -> str:
    """Formats today's date in the target language (French month names for 'fr')."""
    today = today or datetime.date.today()
    if lang == "fr":
        return f"{today.day} {_FRENCH_MONTHS[today.month - 1]} {today.year}"
    return today.strftime("%B %d, %Y")


def escape_latex(text: str) -> str:
    """Escapes special LaTeX characters in plain text strings."""
    if not text:
        return ""
    regex = re.compile("|".join(re.escape(k) for k in LATEX_ESCAPE_MAP.keys()))
    return regex.sub(lambda match: LATEX_ESCAPE_MAP[match.group(0)], text)


def find_pdflatex_binary() -> str:
    """Locates pdflatex from PATH or default Windows installation paths."""
    binary = shutil.which("pdflatex")
    if binary:
        return binary

    local_app_data = os.environ.get("LOCALAPPDATA", "")
    user_miktex = os.path.join(
        local_app_data, "Programs", "MiKTeX", "miktex", "bin", "x64", "pdflatex.exe"
    )
    if os.path.exists(user_miktex):
        return user_miktex

    system_miktex = r"C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe"
    if os.path.exists(system_miktex):
        return system_miktex

    raise EnvironmentError(
        "pdflatex executable was not found. Please ensure MiKTeX or TeX Live is installed."
    )


def render_cv_latex(selection: CVSelection, master_cv: MasterCV) -> str:
    """Generates a polished, ATS-compliant LaTeX CV styled after the reference template."""
    lang = selection.language
    p_info = master_cv.personal_info

    # 1. Header / Preamble (mirrors reference main.tex styling)
    # Build the contact line dynamically so phone/location appear only when present.
    contact_parts = []
    if p_info.phone:
        contact_parts.append(escape_latex(p_info.phone))
    if p_info.location:
        contact_parts.append(escape_latex(p_info.location))
    contact_parts.append(
        rf"\href{{mailto:{escape_latex(p_info.email)}}}{{{escape_latex(p_info.email)}}}"
    )
    if p_info.linkedin:
        contact_parts.append(
            rf"\href{{https://{escape_latex(p_info.linkedin)}}}{{{escape_latex(p_info.linkedin)}}}"
        )
    if p_info.github:
        contact_parts.append(
            rf"\href{{https://{escape_latex(p_info.github)}}}{{{escape_latex(p_info.github)}}}"
        )
    contact_line = r" \ \ $|$ \ \ ".join(contact_parts)

    header = rf"""\documentclass[10pt,a4paper]{{article}}
\usepackage[margin=0.6in]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage[hidelinks]{{hyperref}}
\usepackage{{enumitem}}
\usepackage{{titlesec}}

\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\setlist[itemize]{{leftmargin=1.2em, itemsep=1pt, topsep=2pt, parsep=0pt}}

\titleformat{{\section}}{{\large\bfseries}}{{}}{{0pt}}{{}}[\titlerule]
\titlespacing*{{\section}}{{0pt}}{{8pt}}{{4pt}}

\newcommand{{\role}}[3]{{\textbf{{#1}} \hfill #3 \\ \textit{{#2}}}}

\begin{{document}}

% ---------- HEADER ----------
\begin{{center}}
{{\LARGE \textbf{{{escape_latex(p_info.name)}}}}}\\[3pt]
{{\large {escape_latex(selection.target_role)}}}\\[3pt]
{contact_line}
\end{{center}}
"""

    # 2. Selected Skills (flat, comma-separated flowing line)
    skills_map = {s.id: getattr(s, lang) for s in master_cv.skills}
    chosen_skills = [
        escape_latex(skills_map[sid])
        for sid in selection.skill_ids
        if sid in skills_map
    ]

    skills_title = "Compétences Techniques" if lang == "fr" else "Technical Skills"
    skills_body = ", ".join(chosen_skills)
    skills_section = rf"""
% ---------- SKILLS ----------
\section*{{{skills_title}}}
{skills_body}
"""

    # 3. Selected Experience
    exp_title = "Expérience Professionnelle" if lang == "fr" else "Work Experience"
    exp_lookup = {e.id: e for e in master_cv.experience}
    exp_blocks = []

    for idx, sel_exp in enumerate(selection.experiences):
        if sel_exp.experience_id not in exp_lookup:
            continue
        exp = exp_lookup[sel_exp.experience_id]
        role_title = getattr(exp.role, lang)
        org_line = exp.company
        if exp.location:
            org_line = f"{exp.company}, {exp.location}"
        date_line = f"{exp.start_date} -- {exp.end_date}"

        bullet_lookup = {b.id: getattr(b, lang) for b in exp.bullets}
        bullet_items = []
        for bid in sel_exp.bullet_ids:
            if bid in bullet_lookup:
                bullet_items.append(rf"  \item {escape_latex(bullet_lookup[bid])}")

        bullets_latex = "\n".join(bullet_items)
        sep = "\n\\vspace{3pt}" if idx > 0 else ""
        block = rf"""{sep}
\role{{{escape_latex(role_title)}}}{{{escape_latex(org_line)}}}{{{escape_latex(date_line)}}}
\begin{{itemize}}
{bullets_latex}
\end{{itemize}}"""
        exp_blocks.append(block)

    experience_section = rf"""
% ---------- EXPERIENCE ----------
\section*{{{exp_title}}}
{"".join(exp_blocks)}
"""

    # 4. Selected Projects
    proj_section = ""
    if selection.projects:
        proj_title = "Projets" if lang == "fr" else "Projects"
        proj_lookup = {p.id: p for p in master_cv.projects}
        proj_blocks = []

        for idx, sel_proj in enumerate(selection.projects):
            if sel_proj.project_id not in proj_lookup:
                continue
            proj = proj_lookup[sel_proj.project_id]
            desc = getattr(proj.description, lang)

            # Linked project name if a URL exists, otherwise plain bold name.
            if proj.url:
                name_latex = (
                    rf"\href{{{proj.url}}}{{\textbf{{{escape_latex(proj.name)}}}}}"
                )
            else:
                name_latex = rf"\textbf{{{escape_latex(proj.name)}}}"

            bullet_lookup = {b.id: getattr(b, lang) for b in proj.bullets}
            bullet_items = [
                rf"  \item {escape_latex(bullet_lookup[bid])}"
                for bid in sel_proj.bullet_ids
                if bid in bullet_lookup
            ]
            bullets_latex = ""
            if bullet_items:
                bullets_latex = (
                    "\n\\begin{itemize}\n"
                    + "\n".join(bullet_items)
                    + "\n\\end{itemize}"
                )

            sep = "\n\\vspace{3pt}" if idx > 0 else ""
            p_block = rf"""{sep}
{name_latex} \textit{{({escape_latex(desc)})}}{bullets_latex}"""
            proj_blocks.append(p_block)

        proj_section = rf"""
% ---------- PROJECTS ----------
\section*{{{proj_title}}}
{"".join(proj_blocks)}
"""

        # 5. Education — CORE SECTION: always include every education entry, never
    # drop or subset it. Education is mandatory and must survive any pruning.
    edu_title = "Formation" if lang == "fr" else "Education"
    edu_lookup = {e.id: e for e in master_cv.education}

    # Render all education entries from the Master CV (not just the LLM-selected
    # subset), ordered by the selection when possible, then any remaining ones.
    ordered_edu_ids = [eid for eid in selection.education_ids if eid in edu_lookup]
    ordered_edu_ids += [eid for eid in edu_lookup if eid not in ordered_edu_ids]

    edu_blocks = []
    for eid in ordered_edu_ids:
        edu = edu_lookup[eid]
        degree_name = getattr(edu.degree, lang)
        edu_blocks.append(
            rf"\textbf{{{escape_latex(degree_name)}}}, {escape_latex(edu.institution)} \hfill {escape_latex(edu.graduation_year)}"
        )

    edu_body = "\\\\\n".join(edu_blocks)
    education_section = rf"""
% ---------- EDUCATION ----------
\section*{{{edu_title}}}
{edu_body}
"""

    footer = "\n\\end{document}\n"
    return (
        header
        + experience_section
        + proj_section
        + education_section
        + skills_section
        + footer
    )


def compile_latex(
    tex_code: str, output_dir: str = "output", job_name: str = "cv"
) -> tuple[str, int]:
    """Compiles LaTeX code to PDF using pdflatex. Returns (pdf_path, page_count)."""
    pdflatex_bin = find_pdflatex_binary()

    os.makedirs(output_dir, exist_ok=True)
    tex_path = os.path.join(output_dir, f"{job_name}.tex")
    pdf_path = os.path.join(output_dir, f"{job_name}.pdf")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_code)

    cmd = [
        pdflatex_bin,
        "-interaction=nonstopmode",
        f"-output-directory={output_dir}",
        tex_path,
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    if not os.path.exists(pdf_path):
        raise RuntimeError(
            f"LaTeX compilation failed. Check {os.path.join(output_dir, f'{job_name}.log')}"
        )

    doc = pymupdf.open(pdf_path)
    page_count = doc.page_count
    doc.close()

    return pdf_path, page_count


def compile_to_single_page(
    selection: CVSelection,
    master_cv: MasterCV,
    output_dir: str = "output",
    job_name: str = "tailored_cv",
) -> str:
    """FR-13: Compiles CV and drops lowest-ranked items until it fits on exactly 1 page."""
    current_selection = selection.model_copy(deep=True)
    max_prune_iterations = 8

    for iteration in range(max_prune_iterations):
        tex_code = render_cv_latex(current_selection, master_cv)
        pdf_path, pages = compile_latex(
            tex_code, output_dir=output_dir, job_name=job_name
        )

        if pages == 1:
            print(f"[✓] Successfully compiled 1-page CV: {pdf_path}")
            return pdf_path

        print(
            f"[!] Document overflowed ({pages} pages). Pruning lowest-priority content (Pass {iteration + 1})..."
        )

        pruned = False
        longest_exp = max(
            current_selection.experiences, key=lambda e: len(e.bullet_ids), default=None
        )
        if longest_exp and len(longest_exp.bullet_ids) > 1:
            dropped_bullet = longest_exp.bullet_ids.pop()
            print(
                f"    - Pruned bullet '{dropped_bullet}' from experience '{longest_exp.experience_id}'"
            )
            pruned = True
        elif current_selection.projects:
            dropped_proj = current_selection.projects.pop()
            print(f"    - Pruned project '{dropped_proj.project_id}'")
            pruned = True
        elif len(current_selection.skill_ids) > 6:
            dropped_skill = current_selection.skill_ids.pop()
            print(f"    - Pruned skill '{dropped_skill}'")
            pruned = True

        if not pruned:
            print("[!] Cannot prune further without compromising core sections.")
            return pdf_path

    return pdf_path


def render_letter_latex(letter: CoverLetter, master_cv: MasterCV) -> str:
    """Renders a formal ATS-compliant LaTeX cover letter from flat sentence records."""
    p_info = master_cv.personal_info
    lang = letter.language
    current_date = _localized_date(lang)

    # Localized recipient default and sign-off.
    recipient = "Équipe de recrutement" if lang == "fr" else "Hiring Team"
    default_signoff = "Cordialement" if lang == "fr" else "Sincerely"

    # Build the sender contact line dynamically (phone/location only when present).
    sender_parts = []
    if p_info.phone:
        sender_parts.append(escape_latex(p_info.phone))
    if p_info.location:
        sender_parts.append(escape_latex(p_info.location))
    sender_parts.append(
        rf"\href{{mailto:{p_info.email}}}{{{escape_latex(p_info.email)}}}"
    )
    if p_info.linkedin:
        sender_parts.append(
            rf"\href{{https://{p_info.linkedin}}}{{{escape_latex(p_info.linkedin)}}}"
        )
    if p_info.github:
        sender_parts.append(
            rf"\href{{https://{p_info.github}}}{{{escape_latex(p_info.github)}}}"
        )
    sender_line = " $|$ ".join(sender_parts)

    header = rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[margin=1.0in]{{geometry}}
\usepackage{{hyperref}}
\usepackage{{parskip}}

\hypersetup{{
    colorlinks=true,
    linkcolor=black,
    urlcolor=blue
}}

\pagestyle{{empty}}

\begin{{document}}

\begin{{flushleft}}
    {{\LARGE \textbf{{{escape_latex(p_info.name)}}}}} \\[4pt]
    {sender_line}
\end{{flushleft}}

\vspace{{6pt}}
\hrule
\vspace{{14pt}}

\begin{{flushleft}}
    \textbf{{{escape_latex(current_date)}}} \\[8pt]
    \textbf{{{escape_latex(recipient)}}} \\
    \textbf{{{escape_latex(letter.company)}}}
\end{{flushleft}}

\vspace{{12pt}}
\noindent {escape_latex(letter.salutation)}

"""

    # Group flat sentences by their paragraph number
    para_dict = defaultdict(list)
    for s in letter.sentences:
        if s.text.strip():
            para_dict[s.paragraph].append(s.text.strip())

    body_paragraphs = []
    for p_num in sorted(para_dict.keys()):
        paragraph_text = " ".join(para_dict[p_num])
        # Defensive: strip any greeting the model left at the start of the body,
        # since the salutation is rendered separately above.
        paragraph_text = _strip_leading_greeting(paragraph_text)
        if paragraph_text:
            body_paragraphs.append(escape_latex(paragraph_text))

    body_latex = "\n\n".join(body_paragraphs)

    # Determine the sign-off. Honour a model-provided sign-off only if it's a
    # genuine phrase (not the candidate's name), otherwise use the localized
    # default. Strip any leading sign-off phrase the model embedded in the body.
    clean_signoff = _SIGNOFF_PREFIX_RE.sub("", letter.signoff.strip(), count=1).strip()
    clean_signoff = clean_signoff.split(",")[0].strip()
    if (
        not clean_signoff
        or clean_signoff.lower().endswith(p_info.name.lower())
        or p_info.name.lower() in clean_signoff.lower()
    ):
        clean_signoff = default_signoff

    footer = rf"""

\vspace{{14pt}}
\noindent {escape_latex(clean_signoff)}, \\[24pt]
\textbf{{{escape_latex(p_info.name)}}}

\end{{document}}
"""

    return header + body_latex + footer


def compile_letter_pdf(
    letter: CoverLetter,
    master_cv: MasterCV,
    output_dir: str = "output",
    job_name: str = "cover_letter",
) -> str:
    """Compiles the cover letter into a standalone PDF."""
    tex_code = render_letter_latex(letter, master_cv)
    pdf_path, _ = compile_latex(tex_code, output_dir=output_dir, job_name=job_name)
    return pdf_path
