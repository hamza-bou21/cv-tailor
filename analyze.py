import argparse
import os
import sys
from core import load_master_cv, analyze_job, select_cv_content, generate_cover_letter
from latex_builder import compile_to_single_page, compile_letter_pdf
from db import init_db, log_document


def parse_args():
    parser = argparse.ArgumentParser(
        description="CV Tailor - Job Fit Analyzer & Selection Engine"
    )
    parser.add_argument(
        "--cv",
        default="master_cv.yaml",
        help="Path to master CV YAML (falls back to demo/master_cv.yaml if missing)",
    )
    parser.add_argument(
        "--job", default="demo/posting.txt", help="Path to job posting text file"
    )
    parser.add_argument(
        "--lang",
        choices=["en", "fr"],
        default="en",
        help="Target language for tailored content ('en' or 'fr', default: 'en')",
    )
    parser.add_argument(
        "--letter",
        action="store_true",
        help="Also generate a verified, tailored cover letter",
    )
    parser.add_argument(
        "--company", default="", help="Company name for the audit log (default: empty)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Generate CV/letter even if the fit verdict is SKIP",
    )
    return parser.parse_args()


def main():
    # Ensure UTF-8 output so glyphs (• ✓ ✗ ★) don't crash cp1252 Windows consoles.
    # reconfigure() exists on Python 3.7+; fall back silently on older/odd streams.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass

    args = parse_args()

    # 0. Require a Groq API key before doing any work
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not api_key.strip():
        print(
            "Error: GROQ_API_KEY environment variable is not set.\n"
            "Set it before running, e.g.:\n"
            "  PowerShell:  $env:GROQ_API_KEY = 'gsk_...'\n"
            "  bash/zsh:    export GROQ_API_KEY='gsk_...'",
            file=sys.stderr,
        )
        sys.exit(1)

    # 1. Initialize SQLite database and tables at the very start of execution
    init_db()

    # Load CV profile
    cv_path = args.cv
    try:
        cv = load_master_cv(cv_path)
    except FileNotFoundError:
        print(f"Notice: '{cv_path}' not found. Falling back to 'demo/master_cv.yaml'.")
        cv = load_master_cv("demo/master_cv.yaml")

    # Load Target Job Description
    try:
        with open(args.job, "r", encoding="utf-8") as f:
            job_text = f.read()
    except FileNotFoundError:
        print(f"Error: Job file '{args.job}' not found.", file=sys.stderr)
        sys.exit(1)

    print("Analyzing job fit against Master CV profile...")
    analysis = analyze_job(job_text, cv, api_key=api_key)

    print("\n" + "=" * 44)
    print(f" FIT SCORE: {analysis.fit_score}/100 | VERDICT: {analysis.verdict.upper()}")
    print("=" * 44)

    print("\n[+] Strengths:")
    for s in analysis.strengths:
        print(f"  • {s}")

    print("\n[-] Gaps:")
    for g in analysis.gaps:
        print(f"  • {g}")

    print_covered = ", ".join(analysis.covered_keywords)
    print_missing = ", ".join(analysis.missing_keywords)
    print("\n[✓] Covered Keywords:", print_covered)
    print("[✗] Missing Keywords:", print_missing)

    if analysis.verdict not in ("apply", "stretch") and not args.force:
        print("\n[!] Verdict is SKIP: Skipping CV and letter generation (no --force).")
        return

    if analysis.verdict == "skip" and args.force:
        print("\n[!] Verdict is SKIP, but proceeding due to --force override.")
    print(f"\nSelecting and ranking CV content ({args.lang.upper()})...")

    try:
        selection = select_cv_content(job_text, cv, api_key=api_key, language=args.lang)

        print("\n" + "=" * 44)
        print(f" TARGET ROLE: {selection.target_role}")
        print("=" * 44)

        print(f"\n[★] Selected Skills ({len(selection.skill_ids)}):")
        print("  " + ", ".join(selection.skill_ids))

        print("\n[★] Selected Experience & Ranked Bullets:")
        for exp in selection.experiences:
            bullet_list = ", ".join(exp.bullet_ids) if exp.bullet_ids else "None"
            print(
                f"  • {exp.experience_id} ({len(exp.bullet_ids)} bullets): {bullet_list}"
            )

        if selection.projects:
            print("\n[★] Selected Projects:")
            for proj in selection.projects:
                bullet_list = ", ".join(proj.bullet_ids) if proj.bullet_ids else "None"
                print(
                    f"  • {proj.project_id} ({len(proj.bullet_ids)} bullets): {bullet_list}"
                )

        # 2. Compile CV PDF and log it to SQLite immediately after generation
        print("\nCompiling single-page CV PDF...")
        cv_pdf = compile_to_single_page(
            selection, cv, output_dir="output", job_name="tailored_cv"
        )
        cv_tex = os.path.join("output", "tailored_cv.tex")

        cv_id = log_document(
            job_title=selection.target_role,
            company=args.company,
            fit_score=analysis.fit_score,
            verdict=analysis.verdict,
            pdf_path=cv_pdf,
            tex_path=cv_tex,
            language=args.lang,
            doc_type="cv",
        )
        print(f"[Done] CV PDF: {cv_pdf} (Audit Log ID: {cv_id})")

        # 3. Compile Cover Letter PDF and log it to SQLite
        if args.letter:
            print(
                f"\nGenerating and verifying tailored cover letter ({args.lang.upper()})..."
            )
            letter = generate_cover_letter(
                job_text, cv, api_key=api_key, language=args.lang
            )

            if letter.verification_passed:
                print("[✓] Cover letter passed deterministic honesty verification.")
            else:
                print(
                    f"[!] Cover letter generated with warnings: {letter.verification_warnings}"
                )

            letter_pdf = compile_letter_pdf(
                letter, cv, output_dir="output", job_name="cover_letter"
            )
            letter_tex = os.path.join("output", "cover_letter.tex")

            warnings_str = (
                ", ".join(letter.verification_warnings)
                if letter.verification_warnings
                else None
            )
            letter_id = log_document(
                job_title=letter.target_role,
                company=letter.company or args.company,
                fit_score=analysis.fit_score,
                verdict=analysis.verdict,
                pdf_path=letter_pdf,
                tex_path=letter_tex,
                language=args.lang,
                doc_type="cover_letter",
                warnings=warnings_str,
            )
            print(f"[Done] Cover Letter PDF: {letter_pdf} (Audit Log ID: {letter_id})")
    except Exception as e:
        print(f"\n[Error] Generation failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
