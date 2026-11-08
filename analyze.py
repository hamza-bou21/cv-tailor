import argparse
import sys
from core import load_master_cv, analyze_job, select_cv_content
from latex_builder import compile_to_single_page

def parse_args():
    parser = argparse.ArgumentParser(description="CV Tailor - Job Fit Analyzer & Selection Engine")
    parser.add_argument(
        "--cv", 
        default="master_cv.yaml", 
        help="Path to master CV YAML (falls back to demo/master_cv.yaml if missing)"
    )
    parser.add_argument(
        "--job", 
        default="demo/posting.txt", 
        help="Path to job posting text file"
    )
    parser.add_argument(
        "--lang",
        choices=["en", "fr"],
        default="en",
        help="Target language for tailored content ('en' or 'fr', default: 'en')"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    cv_path = args.cv
    try:
        cv = load_master_cv(cv_path)
    except FileNotFoundError:
        print(f"Notice: '{cv_path}' not found. Falling back to 'demo/master_cv.yaml'.")
        cv = load_master_cv("demo/master_cv.yaml")

    try:
        with open(args.job, "r", encoding="utf-8") as f:
            job_text = f.read()
    except FileNotFoundError:
        print(f"Error: Job file '{args.job}' not found.", file=sys.stderr)
        sys.exit(1)

    print("Analyzing job fit against Master CV profile...")
    analysis = analyze_job(job_text, cv)

    print("\n" + "=" * 44)
    print(f" FIT SCORE: {analysis.fit_score}/100 | VERDICT: {analysis.verdict.upper()}")
    print("=" * 44)

    print("\n[+] Strengths:")
    for s in analysis.strengths:
        print(f"  • {s}")

    print("\n[-] Gaps:")
    for g in analysis.gaps:
        print(f"  • {g}")

    print("\n[✓] Covered Keywords:", ", ".join(analysis.covered_keywords))
    print("[✗] Missing Keywords:", ", ".join(analysis.missing_keywords))

    if analysis.verdict in ("apply", "stretch"):
        print(f"\nSelecting and ranking CV content ({args.lang.upper()})...")
        selection = select_cv_content(job_text, cv, language=args.lang)

        print("\n" + "=" * 44)
        print(f" TARGET ROLE: {selection.target_role}")
        print("=" * 44)

        print(f"\n[★] Selected Skills ({len(selection.skill_ids)}):")
        print("  " + ", ".join(selection.skill_ids))

        print("\n[★] Selected Experience & Ranked Bullets:")
        for exp in selection.experiences:
            bullet_list = ", ".join(exp.bullet_ids) if exp.bullet_ids else "None"
            print(f"  • {exp.experience_id} ({len(exp.bullet_ids)} bullets): {bullet_list}")

        if selection.projects:
            print("\n[★] Selected Projects:")
            for proj in selection.projects:
                bullet_list = ", ".join(proj.bullet_ids) if proj.bullet_ids else "None"
                print(f"  • {proj.project_id} ({len(proj.bullet_ids)} bullets): {bullet_list}")

        # Compile PDF via the single-page constraint loop
        print("\nCompiling single-page PDF...")
        pdf_path = compile_to_single_page(selection, cv, output_dir="output", job_name="tailored_cv")
        print(f"[Done] Ready at: {pdf_path}")
    else:
        print("\n[!] Verdict is SKIP: Skipping CV content selection and compilation.")

if __name__ == "__main__":
    main()