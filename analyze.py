import argparse
import sys
from core import load_master_cv, analyze_job

def parse_args():
    parser = argparse.ArgumentParser(description="CV Tailor - Job Fit Analyzer")
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
    return parser.parse_args()

def main():
    args = parse_args()

    # Fallback to demo CV if personal CV is not found
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

if __name__ == "__main__":
    main()