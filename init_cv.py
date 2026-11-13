import os
import sys

from core import build_master_cv_from_pdf, save_master_cv


def extract_cv_to_yaml(
    pdf_path: str, api_key: str, output_path: str = "master_cv.yaml"
) -> None:
    """Extracts a PDF CV via the shared core pipeline and saves it as YAML."""
    print(f"Reading {pdf_path}...")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    print("Extracting text, hyperlinks, and calling Groq for structured extraction...")
    validated_cv, warnings = build_master_cv_from_pdf(pdf_bytes, api_key=api_key)

    if warnings:
        print("\n[!] Grounding warnings (review the extracted YAML):")
        for w in warnings:
            print(f"    - {w}")

    print(f"Validation successful! Writing to {output_path}...")
    save_master_cv(validated_cv, output_path)

    print("Done! Your new Master CV database is ready.")


if __name__ == "__main__":
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

    if len(sys.argv) < 2:
        print("Usage: python init_cv.py <path_to_your_cv.pdf>")
        sys.exit(1)

    pdf_file_path = sys.argv[1]

    if not os.path.exists(pdf_file_path):
        print(f"Error: Could not find the file '{pdf_file_path}'")
        sys.exit(1)

    extract_cv_to_yaml(pdf_file_path, api_key)
