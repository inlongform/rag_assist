import os
import uuid
import json
import re
import unicodedata
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------------------
# Config
# ------------------------------
PATH_LIST = [
    Path('docs/lab_docs'),
    Path('docs/lab_examples_combined'),
    Path('docs/pxr_docs'),
    Path('docs/repl_docs'),
    Path('docs/sim_docs')
]

OUTPUT_DIR = Path("docs/processed/chunks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

IGNORE_PATTERNS = ["_images", "assets", "node_modules", ".git"]

# ------------------------------
# Pydantic Schema
# ------------------------------
class DocChunk(BaseModel):
    id: str
    source: str
    section: Optional[str] = None
    content: str
    code: Optional[str] = None
    tags: Optional[List[str]] = []

# ------------------------------
# Helpers
# ------------------------------
CODE_BLOCK_PATTERN = r"```[\s\S]*?```"

def clean_text(text: str) -> str:
    """Normalize text: remove excessive whitespace, redundant blank lines, and common artifacts."""
    # Remove 'Skip to...' headers
    text = re.sub(r'\[Skip to.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Skip to main content', '', text, flags=re.IGNORECASE)
    # Remove multi-line "(Back to top ... " blocks
    text = re.sub(r'^\(\s*\nBack to top.*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    # Remove lines containing only *, -, or whitespace (empty bullets)
    text = re.sub(r'^[\*\-\s]+$', '', text, flags=re.MULTILINE)
    # Remove trailing spaces/tabs
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    # Remove any line containing "Copy to clipboard"
    text = re.sub(r'^.*Copy to clipboard.*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    # Collapse multiple newlines
    text = re.sub(r'\s+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)
    return text.strip()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n## ", "\n### ", "\n", " ", ""]
)

def split_preserving_code(text: str):
    """Split into text and code segments, preserving code blocks as atomic chunks."""
    parts = []
    last_end = 0

    for match in re.finditer(CODE_BLOCK_PATTERN, text):
        start, end = match.span()
        if start > last_end:
            parts.append({"type": "text", "content": text[last_end:start]})
        parts.append({"type": "code", "content": match.group(0)})
        last_end = end

    if last_end < len(text):
        parts.append({"type": "text", "content": text[last_end:]})

    return parts

def build_chunks(text: str, source: str):
    segments = split_preserving_code(text)
    chunks = []

    for segment in segments:
        if segment["type"] == "code":
            chunks.append(
                DocChunk(
                    id=str(uuid.uuid4()),
                    source=source,
                    content=segment["content"],
                    code=segment["content"]
                ).model_dump()
            )
        else:
            for chunk_text in splitter.split_text(segment["content"]):
                chunks.append(
                    DocChunk(
                        id=str(uuid.uuid4()),
                        source=source,
                        content=chunk_text
                    ).model_dump()
                )
    return chunks

def process_file(path: Path):
    if any(skip in str(path) for skip in IGNORE_PATTERNS):
        return []

    try:
        raw = path.read_text(errors="ignore")
    except:
        return []

    segments = split_preserving_code(raw)
    cleaned_text = ""

    for seg in segments:
        if seg["type"] == "code":
            cleaned_text += seg["content"] + "\n"
        else:
            cleaned_text += clean_text(seg["content"]) + "\n"

    return build_chunks(cleaned_text, str(path))

def process_all():
    for p in PATH_LIST:
        for root, _, files in os.walk(p):
            for file in files:
                if not file.lower().endswith(".md"):
                    continue
                fp = Path(root) / file
                file_chunks = process_file(fp)
                if not file_chunks:
                    continue

                out_file = OUTPUT_DIR / (fp.stem + "_chunks.jsonl")
                with out_file.open("w", encoding="utf-8") as f:
                    for chunk in file_chunks:
                        f.write(json.dumps(chunk) + "\n")

                print(f"✔ Processed: {file}, chunks: {len(file_chunks)}")

if __name__ == "__main__":
    process_all()
