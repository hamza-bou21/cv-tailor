import os
import re
import shutil
import subprocess
import pymupdf
from typing import Optional
from models import MasterCV, CVSelection

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
    user_miktex = os.path.join(local_app_data, "Programs", "MiKTeX", "miktex", "bin", "x64", "pdflatex.exe")
    if os.path.exists(user_miktex):
        return user_miktex

    system_miktex = r"C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe"
    if os.path.exists(system_miktex):
        return system_miktex

    raise EnvironmentError("pdflatex executable was not found. Please ensure MiKTeX or TeX Live is installed.")

def render_cv_latex(selection: CVSelection, master_cv: MasterCV) -> str:
    """Generates clean, ATS-compliant LaTeX code from selected CV items."""
    lang = selection.language
    p_info = master_cv.personal_info
    
    # 1. Header Section
    header = rf"""\documentclass[10pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[margin=0.6in]{{geometry}}
\usepackage{{enumitem}}
\usepackage{{hyperref}}
\usepackage{{titlesec}}

\hypersetup{{
    colorlinks=true,
    linkcolor=black,
    urlcolor=blue
}}

\titleformat{{\section}}{{\large\bfseries\uppercase}}{{}}{{0em}}{{}}[\titlerule]
\titlespacing*{{\section}}{{0pt}}{{8pt}}{{4pt}}
\setlist[itemize]{{leftmargin=1.2em, itemsep=1pt, topsep=2pt, parsep=0pt}}
\pagestyle{{empty}}

\begin{{document}}

\begin{{center}}
    {{\LARGE \textbf{{{escape_latex(p_info.name)}}}}} \\[3pt]
    \href{{mailto:{p_info.email}}}{{{escape_latex(p_info.email)}}} $|$ 
    \href{{https://{p_info.linkedin}}}{{{escape_latex(p_info.linkedin)}}} $|$ 
    \href{{https://{p_info.github}}}{{{escape_latex(p_info.github)}}}
\end{{center}}
\vspace{{-8pt}}
"""

    # 2. Selected Skills Section
    skills_map = {s.id: getattr(s, lang) for s in master_cv.skills}
    chosen_skills = [escape_latex(skills_map[sid]) for sid in selection.skill_ids if sid in skills_map]
    
    skills_title = "Compétences Techniques" if lang == "fr" else "Technical Skills"
    skills_section = rf"""
\section*{{{skills_title}}}
\noindent {", ".join(chosen_skills)}
"""

    # 3. Selected Experience Section
    exp_title = "Expérience Professionnelle" if lang == "fr" else "Work Experience"
    exp_lookup = {e.id: e for e in master_cv.experience}
    exp_blocks = []

    for sel_exp in selection.experiences:
        if sel_exp.experience_id not in exp_lookup:
            continue
        exp = exp_lookup[sel_exp.experience_id]
        role_title = getattr(exp.role, lang)
        
        bullet_lookup = {b.id: getattr(b, lang) for b in exp.bullets}
        bullet_items = []
        for bid in sel_exp.bullet_ids:
            if bid in bullet_lookup:
                bullet_items.append(rf"\item {escape_latex(bullet_lookup[bid])}")

        bullets_latex = "\n        ".join(bullet_items)
        block = rf"""
\noindent \textbf{{{escape_latex(role_title)}}} \hfill \textbf{{{escape_latex(exp.company)}}} \\
\textit{{{escape_latex(exp.start_date)} -- {escape_latex(exp.end_date)}}} \hfill \textit{{{escape_latex(exp.location)}}}
\begin{{itemize}}
        {bullets_latex}
\end{{itemize}}
\vspace{{2pt}}"""
        exp_blocks.append(block)

    experience_section = rf"""
\section*{{{exp_title}}}
{"".join(exp_blocks)}
"""

    # 4. Selected Projects Section
    proj_section = ""
    if selection.projects:
        proj_title = "Projets" if lang == "fr" else "Projects"
        proj_lookup = {p.id: p for p in master_cv.projects}
        proj_blocks = []

        for sel_proj in selection.projects:
            if sel_proj.project_id not in proj_lookup:
                continue
            proj = proj_lookup[sel_proj.project_id]
            desc = getattr(proj.description, lang)
            
            bullet_lookup = {b.id: getattr(b, lang) for b in proj.bullets}
            bullet_items = [
                rf"\item {escape_latex(bullet_lookup[bid])}"
                for bid in sel_proj.bullet_ids if bid in bullet_lookup
            ]
            bullets_latex = f"\n\\begin{{itemize}}\n        {''.join(bullet_items)}\n\\end{{itemize}}" if bullet_items else ""

            p_block = rf"""
\noindent \textbf{{{escape_latex(proj.name)}}} $|$ \textit{{{escape_latex(desc)}}} {bullets_latex}
\vspace{{2pt}}"""
            proj_blocks.append(p_block)

        proj_section = rf"""
\section*{{{proj_title}}}
{"".join(proj_blocks)}
"""

    # 5. Education Section
    edu_title = "Formation" if lang == "fr" else "Education"
    edu_blocks = []
    for edu in master_cv.education:
        degree_name = getattr(edu.degree, lang)
        edu_blocks.append(
            rf"\noindent \textbf{{{escape_latex(edu.institution)}}} \hfill {escape_latex(edu.graduation_year)} \\"
            rf"\textit{{{escape_latex(degree_name)}}}"
        )

    # Join outside of the f-string curly braces to comply with Python 3.11 syntax
    edu_body = "\n\\vspace{2pt}\n".join(edu_blocks)
    education_section = rf"""
\section*{{{edu_title}}}
{edu_body}
"""

    footer = r"\end{document}"
    return header + skills_section + experience_section + proj_section + education_section + footer

def compile_latex(tex_code: str, output_dir: str = "output", job_name: str = "cv") -> tuple[str, int]:
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
        tex_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    if not os.path.exists(pdf_path):
        raise RuntimeError(f"LaTeX compilation failed. Check {os.path.join(output_dir, f'{job_name}.log')}")

    doc = pymupdf.open(pdf_path)
    page_count = doc.page_count
    doc.close()

    return pdf_path, page_count

def compile_to_single_page(
    selection: CVSelection, 
    master_cv: MasterCV, 
    output_dir: str = "output",
    job_name: str = "tailored_cv"
) -> str:
    """FR-13: Compiles CV and drops lowest-ranked items until it fits on exactly 1 page."""
    current_selection = selection.model_copy(deep=True)
    max_prune_iterations = 8

    for iteration in range(max_prune_iterations):
        tex_code = render_cv_latex(current_selection, master_cv)
        pdf_path, pages = compile_latex(tex_code, output_dir=output_dir, job_name=job_name)

        if pages == 1:
            print(f"[✓] Successfully compiled 1-page CV: {pdf_path}")
            return pdf_path

        print(f"[!] Document overflowed ({pages} pages). Pruning lowest-priority content (Pass {iteration + 1})...")

        pruned = False
        longest_exp = max(current_selection.experiences, key=lambda e: len(e.bullet_ids), default=None)
        if longest_exp and len(longest_exp.bullet_ids) > 1:
            dropped_bullet = longest_exp.bullet_ids.pop()
            print(f"    - Pruned bullet '{dropped_bullet}' from experience '{longest_exp.experience_id}'")
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