import os
import json
import yaml
import pymupdf as fitz
from dotenv import load_dotenv
from groq import Groq
from models import MasterCV
import sys


load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_pdf_data(pdf_path: str) -> str:
    """Reads a PDF and extracts both text and embedded hyperlinks."""
    print(f"Reading {pdf_path}...")
    doc = fitz.open(pdf_path)
    
    full_text = ""
    all_links = set()
    
    for page in doc:
        full_text += page.get_text()
        
        # Extract hidden hyperlinks from the page
        links = page.get_links()
        for link in links:
            if 'uri' in link:
                all_links.add(link['uri'])
                
    doc.close()
    
    # Combine the text and the links so the AI has everything
    combined_content = f"--- CV TEXT ---\n{full_text}\n\n--- EXTRACTED HYPERLINKS ---\n"
    for link in all_links:
        combined_content += f"- {link}\n"
        
    return combined_content

def extract_cv_to_yaml(pdf_path: str, output_path: str = "master_cv.yaml"):
    raw_content = extract_pdf_data(pdf_path)
    print("Sending PDF text and links to Groq for extraction...")
    
    schema = MasterCV.model_json_schema()
    
    prompt = f"""
    You are an expert CV data extractor. Parse the following CV text and the list of extracted 
    hyperlinks, and convert them STRICTLY into the provided JSON schema. 
    
    RULES:
    1. Match the extracted hyperlinks to the correct fields (LinkedIn, GitHub, Portfolio, Project URLs).
    2. Generate unique IDs for all skills, experience bullets, projects, and education.
    3. For any field requiring `BilingualText` (en/fr), provide the original English and a professional French translation.
    4. Extract concrete metrics (years of experience, etc.) into the `facts` block.
    
    RAW DATA:
    {raw_content}
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": f"You are a helpful assistant that outputs JSON matching this schema: {json.dumps(schema)}"},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    
    data_dict = json.loads(response.choices[0].message.content)
    
    print("Validating AI output against Pydantic models...")
    validated_cv = MasterCV(**data_dict)
    
    print(f"Validation successful! Writing to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(validated_cv.model_dump(exclude_none=True), f, allow_unicode=True, sort_keys=False)
        
    print("Done! Your new Master CV database is ready.")


if __name__ == "__main__":
    # Check if the user provided a file path in the terminal
    if len(sys.argv) < 2:
        print("Usage: python init_cv.py <path_to_your_cv.pdf>")
        sys.exit(1)
        
    # Get the file path from the terminal argument
    pdf_file_path = sys.argv[1]
    
    # Check if the file actually exists before trying to read it
    if not os.path.exists(pdf_file_path):
        print(f"Error: Could not find the file '{pdf_file_path}'")
        sys.exit(1)
        
    # Run the extraction
    extract_cv_to_yaml(pdf_file_path)