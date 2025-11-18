import re
from pathlib import Path

import unicodedata

# Input directories containing your docs
PATH_LIST = [
    Path('docs/lab_docs'),
    Path('docs/lab_examples_combined'),
    Path('docs/pxr_docs'),
    Path('docs/repl_docs'),
    Path('docs/sim_docs')
]

# Output directory for chunks
OUTPUT_DIR = Path("docs/processed/chunks")
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_CHARS = 8192
OVERLAP = 1024

# Config
MAX_CHUNK_LINES = 80  # max number of lines per chunk

def clean_doc_text(text: str) -> str:
    # Remove 'Skip to...' headers and main content markers
    text = re.sub(r'\[Skip to.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Skip to main content', '', text, flags=re.IGNORECASE)

    # Remove multi-line "(Back to top ... " blocks
    text = re.sub(r'^\(\s*\nBack to top.*$', '', text, flags=re.IGNORECASE | re.MULTILINE)

    # Remove lines containing only *, -, or whitespace (empty bullets)
    text = re.sub(r'^[\*\-\s]+$', '', text, flags=re.MULTILINE)

    # Remove trailing spaces/tabs
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)

    # Collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove any line containing "Copy to clipboard"
    text = re.sub(r'^.*Copy to clipboard.*$', '', text, flags=re.IGNORECASE | re.MULTILINE)

    # Remove leading line numbers from code lines (e.g., "156def compute_rewards(...")
    text = re.sub(r'^\s*\d+(?=\S)', '', text, flags=re.MULTILINE)

    # Replace Markdown links [text](url) with just text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Remove orphan parentheses on lines by themselves
    text = re.sub(r'^[\(\)]\s*$', '', text, flags=re.MULTILINE)

    # Fix dangling or repeated parentheses and pipes in table-like content
    # Replace sequences like "(0.01, 1000000|" with just "(0.01, 1000000)"
    text = re.sub(r'\(([^)]+)\|+', r'(\1)', text)

    # Collapse repeated pipe symbols into a single one
    text = re.sub(r'\|{2,}', '|', text)

    # Optional: remove leading/trailing pipe on each line
    text = re.sub(r'^\|+|\|+$', '', text, flags=re.MULTILINE)

    # Normalize Unicode (fix mis-encoded characters)
    text = unicodedata.normalize("NFKC", text)

    return text.strip()

def chunk_text(text: str, max_chars=MAX_CHARS, overlap=OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + max_chars, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == text_len:
            break
        start = end - overlap  # overlap for context

    return chunks

def process_file(file: Path, base: Path):
    rel_path = file.relative_to(base)
    text = file.read_text(encoding="utf-8", errors="ignore")
    text = clean_doc_text(text)
    chunks = chunk_text(text)
    
    for i, chunk in enumerate(chunks):
        chunk_file = OUTPUT_DIR / rel_path.parent / f"{rel_path.stem}_chunk{i+1}.md"
        chunk_file.parent.mkdir(parents=True, exist_ok=True)
        chunk_file.write_text(chunk, encoding="utf-8")
        print(f"✔ Saved chunk: {chunk_file}")

def run():
    for d in PATH_LIST:
        for file in d.rglob("*.md"):
            process_file(file, d)

if __name__ == "__main__":
    run()