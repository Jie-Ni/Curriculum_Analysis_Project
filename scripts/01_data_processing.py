import json
import re
import os
from pathlib import Path
import config


def clean_text(text):
    """
    Cleans raw text by removing excessive whitespace, special characters,
    and non-printable characters.
    """
    # Remove HTML tags if any (basic regex)
    text = re.sub(r'<[^>]+>', '', text)
    # Replace multiple newlines/tabs with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove non-printable characters
    text = "".join([c for c in text if c.isprintable()])
    return text.strip()


def process_raw_data():
    """
    Reads raw text files from data/raw/{University}/ and converts them
    into a standardized JSONL format.
    """
    print(f"Processing raw data from {config.RAW_DATA_DIR}...")

    # If no raw data exists, generate dummy data for demonstration
    if not any(config.RAW_DATA_DIR.iterdir()):
        print("No raw data found. Generating synthetic demo data...")
        generate_demo_data()
        return

    documents = []

    # Iterate through university folders in data/raw/
    # Expected structure: data/raw/NJU/course1.txt
    for uni_folder in config.RAW_DATA_DIR.iterdir():
        if uni_folder.is_dir():
            university_name = uni_folder.name  # e.g., "NJU"
            print(f"  Processing {university_name}...")

            for file_path in uni_folder.glob("*.txt"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        raw_content = f.read()

                    cleaned_content = clean_text(raw_content)

                    if len(cleaned_content) > 50:  # Skip empty/short files
                        doc = {
                            "university": university_name,
                            "filename": file_path.name,
                            "content": cleaned_content,
                            "length": len(cleaned_content)
                        }
                        documents.append(doc)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    # Save to JSONL
    output_path = config.INPUT_CORPUS
    print(f"Saving {len(documents)} documents to {output_path}...")

    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')

    print("✅ Data processing complete.")


def generate_demo_data():
    """Generates synthetic data if no raw files are present."""
    demo_docs = []
    unis = ["NJU", "PKU", "Stanford", "MIT", "Heidelberg", "LMU"]

    for uni in unis:
        # Create dummy content
        content = f"This is a sample curriculum document for {uni}. " \
                  f"It includes core modules on Computer Science and Social Theory. " \
                  f"Students must complete 120 credits. Assessment includes exams and projects."

        for i in range(50):  # Generate 50 dummy docs per uni
            demo_docs.append({
                "university": uni,
                "filename": f"demo_{uni}_{i}.txt",
                "content": content,
                "length": len(content)
            })

    with open(config.INPUT_CORPUS, 'w', encoding='utf-8') as f:
        for doc in demo_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    print("✅ Synthetic demo data generated.")


if __name__ == "__main__":
    process_raw_data()