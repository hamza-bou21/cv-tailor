from core import load_master_cv, analyze_job

sample_job = """
Senior Software Engineer - Python & Distributed Systems
Company: CloudScale
Requirements:
- 5+ years experience building backend services in Python.
- Proven experience with microservices, AWS (EC2, S3, RDS), and Redis caching.
- Strong team leadership or tech lead experience.
- Experience with high-throughput distributed systems handling >5,000 RPS.
- Nice to have: Rust experience, Kubernetes.
"""

def run():
    print("Loading master_cv.yaml...")
    cv = load_master_cv()
    
    print("Running Fit Analysis on sample job posting...")
    analysis = analyze_job(sample_job, cv)
    
    print("\n" + "="*40)
    print(f"FIT SCORE: {analysis.fit_score}/100 | VERDICT: {analysis.verdict.upper()}")
    print("="*40)
    
    print("\n[+] Strengths:")
    for s in analysis.strengths:
        print(f"  • {s}")
        
    print("\n[-] Gaps:")
    for g in analysis.gaps:
        print(f"  • {g}")
        
    print("\n[✓] Covered Keywords:", ", ".join(analysis.covered_keywords))
    print("[✗] Missing Keywords:", ", ".join(analysis.missing_keywords))

if __name__ == "__main__":
    run()