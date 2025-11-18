import re
from pathlib import Path

INPUT_DIR = Path("docs/processed/chunks")
OUTPUT_DIR = Path("docs/processed/cleaned_chunks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def remove_dangling_parentheses(text: str) -> str:
    """
    Remove unmatched or dangling parentheses in a text.
    This preserves content inside balanced parentheses.
    """
    # Remove dangling '(' at the end of a line
    text = re.sub(r'\(\s*$', '', text, flags=re.MULTILINE)
    # Remove dangling ')' at the start of a line
    text = re.sub(r'^\s*\)', '', text, flags=re.MULTILINE)
    
    # Remove any standalone dangling parentheses
    text = re.sub(r'\(\)', '', text)
    
    # Optional: collapse multiple consecutive dangling )
    text = re.sub(r'\)+', ')', text)
    return text

def process_file(file: Path, base: Path):
    rel_path = file.relative_to(base)
    out_file = OUTPUT_DIR / rel_path
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    text = file.read_text(encoding='utf-8', errors='ignore')
    cleaned = remove_dangling_parentheses(text)
    out_file.write_text(cleaned, encoding='utf-8')
    print(f"✔ Cleaned dangling parentheses: {file} → {out_file}")

def run():
    for file in INPUT_DIR.rglob("*.md"):
        process_file(file, INPUT_DIR)

if __name__ == "__main__":
    run()
