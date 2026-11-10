import os
import sys

import pymupdf

from core import extract_text_from_pdf, parse_pdf_to_master_cv, save_master_cv


def extract_hyperlinks(pdf_path: str) -> list[str]:
    """Extracts embedded hyperlink URIs from a PDF (unique, order-preserving)."""
    links: list[str] = []
    seen: set[str] = set()
    doc = pymupdf.open(pdf_path)
    try:
        for page in doc:
            for link in page.get_links():
                uri = link.get("uri")
                if uri and uri not in seen:
                    seen.add(uri)
                    links.append(uri)
    finally:
        doc.close()
    return links


def extract_cv_to_yaml(
    pdf_path: str, api_key: str, output_path: str = "master_cv.yaml"
) -> None:
    """Extracts a PDF CV, builds a MasterCV via the shared core pipeline, and saves it as YAML."""
    print(f"Reading {pdf_path}...")
    raw_text = extract_text_from_pdf(pdf_path)
    hyperlinks = extract_hyperlinks(pdf_path)

    combined_content = (
        raw_text if raw_text.strip() else "(no extractable text found in PDF)"
    )

    if hyperlinks:
        link_block = "\n".join(f"- {link}" for link in hyperlinks)
        combined_content += f"\n\n--- EXTRACTED HYPERLINKS ---\n{link_block}"

    if len(raw_text.strip()) < 50:
        print("Warning: Very little text was extracted. The CV may be a scanned image.")

    print("Sending PDF text and links to Groq for extraction...")
    validated_cv = parse_pdf_to_master_cv(combined_content, api_key=api_key)

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
