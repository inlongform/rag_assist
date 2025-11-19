#!/usr/bin/env python3
"""
embedding_preprocessor.py

- Preserves code blocks as atomic chunks
- Cleans text (no destructive regex)
- Extracts header (H1..H6) context and stores it in `section`
- Adds tags and filename/module metadata
- Writes one JSONL file per input .md file (OPTION A)
- Output schema matches Qdrant-friendly payloads

Usage:
    python embedding_preprocessor.py
"""

import os
import uuid
import json
import re
import unicodedata
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

# Optional: progress printing
from pprint import pformat

# You used this splitter previously; keep it (install langchain_text_splitters if needed)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------------------
# Config (adjust to taste)
# ------------------------------
PATH_LIST = [
    Path("docs/lab_docs_cleaned"),
    Path("docs/lab_examples_combined"),
    Path("docs/pxr_docs"),
    Path("docs/repl_docs"),
    Path("docs/sim_docs"),
]

OUTPUT_DIR = Path("docs/processed/chunks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

IGNORE_PATTERNS = ["_images", "assets", "node_modules", ".git"]

# Header regex (MD H1..H6)
HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.*)$', flags=re.MULTILINE)

# Code block pattern - preserves fenced code blocks (```lang ... ```)
CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", flags=re.MULTILINE)

# ------------------------------
# Pydantic Schema
# ------------------------------
class DocChunk(BaseModel):
    id: str
    source: str
    filename: Optional[str] = None
    module: Optional[str] = None
    section: Optional[str] = None
    content: str
    code: Optional[str] = None
    tags: Optional[List[str]] = []
    meta: Optional[Dict[str, Any]] = {}

    def to_payload(self) -> Dict[str, Any]:
        # This method returns a payload ready for JSONL -> Qdrant ingestion (embeddings separate)
        return {
            "id": self.id,
            "source": self.source,
            "filename": self.filename,
            "module": self.module,
            "section": self.section,
            "content": self.content,
            "code": self.code,
            "tags": self.tags or [],
            "meta": self.meta or {},
        }

# ------------------------------
# Helpers
# ------------------------------
def clean_text(text: str) -> str:
    """Normalize text: remove excessive whitespace, redundant blank lines, and common artifacts.
       This avoids destructive regexes and keeps content meaningful.
    """
    if not text:
        return ""

    # Remove 'Skip to...' headers and common nav artifacts
    text = re.sub(r'\[Skip to.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Skip to main content', '', text, flags=re.IGNORECASE)

    # Remove obvious "Back to top" blocks and other nav fragments
    text = re.sub(r'^\(\s*\nBack to top.*$', '', text, flags=re.IGNORECASE | re.MULTILINE)

    # Remove repeated copyright / author lines
    text = re.sub(r'^(By The Isaac Lab Project Developers\.)$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'^(© Copyright.*)$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'^Last updated on .*$', '', text, flags=re.IGNORECASE | re.MULTILINE)

    # Remove SPDX / License blocks (common header forms)
    text = re.sub(
        r'# SPDX-FileCopyrightText:.*?^# limitations under the License\.',
        '',
        text,
        flags=re.DOTALL | re.MULTILINE | re.IGNORECASE
    )
    text = re.sub(
        r'SPDX-License-Identifier:.*?$',
        '',
        text,
        flags=re.MULTILINE | re.IGNORECASE
    )

    # Remove lines which are only bullets / separators
    text = re.sub(r'^[\-\*\s]{1,}$', '', text, flags=re.MULTILINE)

    # Trim trailing whitespace on each line
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)

    # Collapse many blank lines into two
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    return text.strip()


def split_preserving_code(text: str):
    """
    Splits input text into a list of segments where each segment is either:
    {"type": "code", "content": "```...```"} or {"type": "text", "content": "..."}
    Code blocks are preserved atomically.
    """
    parts = []
    last_end = 0
    for m in CODE_BLOCK_PATTERN.finditer(text):
        start, end = m.span()
        if start > last_end:
            parts.append({"type": "text", "content": text[last_end:start]})
        parts.append({"type": "code", "content": m.group(0)})
        last_end = end
    if last_end < len(text):
        parts.append({"type": "text", "content": text[last_end:]})
    return parts


def split_text_by_headers(text: str):
    """
    Given a text block (no fenced code blocks inside), split it into
    sub-blocks using markdown headers as boundaries. Returns a list of
    {"section": "H1 / H2 / ...", "content": "..."} preserving section context.
    """
    out = []
    lines = text.splitlines(keepends=True)
    current_section_stack: List[str] = []
    buffer_lines: List[str] = []

    def flush_buffer():
        if buffer_lines:
            out.append({
                "section": " / ".join(current_section_stack) if current_section_stack else None,
                "content": "".join(buffer_lines).strip()
            })
            buffer_lines.clear()

    for line in lines:
        header_match = HEADER_PATTERN.match(line)
        if header_match:
            # flush existing buffer (text before this header belongs to previous section)
            flush_buffer()
            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            # Maintain stack by level
            current_section_stack = current_section_stack[: level - 1]  # type: ignore
            current_section_stack.append(title)
            # header line itself can be included or not; we include it as context in the next buffer
            buffer_lines.append(line)
        else:
            buffer_lines.append(line)

    flush_buffer()
    return out


# Create the text splitter instance (same as you had)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n## ", "\n### ", "\n", " ", ""],
)


def build_chunks_from_text_segment(text_segment: str, source: str, filename: str, module: Optional[str], section: Optional[str]):
    """
    Use splitter to build chunk payloads from a plain text segment (no fenced code).
    Attach section and metadata to each resulting chunk.
    """
    chunks = []
    # If the segment is empty, nothing to do
    if not text_segment or not text_segment.strip():
        return chunks

    # Split text into smaller chunks
    for chunk_text in splitter.split_text(text_segment):
        chunk = DocChunk(
            id=str(uuid.uuid4()),
            source=source,
            filename=filename,
            module=module,
            section=section,
            content=chunk_text.strip(),
            code=None,
            tags=[filename] if filename else [],
            meta={"origin": source},
        )
        chunks.append(chunk)
    return chunks


def build_chunks(text: str, source: str):
    """
    Given full file text (with code and text), produce a list of DocChunk instances
    preserving code blocks as atomic chunks and splitting text by headers.
    """
    segments = split_preserving_code(text)
    all_chunks: List[DocChunk] = []

    # derive filename and module from source path
    try:
        p = Path(source)
        filename = p.name
        module = str(p.parent).replace(os.sep, ".") if p.parent else None
    except Exception:
        filename = source
        module = None

    for seg in segments:
        if seg["type"] == "code":
            # Preserve code blocks as single chunks
            code_content = seg["content"].strip()
            c = DocChunk(
                id=str(uuid.uuid4()),
                source=source,
                filename=filename,
                module=module,
                section=None,
                content=code_content,
                code=code_content,
                tags=[filename] if filename else [],
                meta={"origin": source, "is_code": True},
            )
            all_chunks.append(c)
        else:
            # For text segments, further split by headers and then chunk via tokenizer
            subsegments = split_text_by_headers(seg["content"])
            for sub in subsegments:
                sec = sub.get("section")
                content = clean_text(sub.get("content", ""))
                if not content:
                    continue
                text_chunks = build_chunks_from_text_segment(content, source, filename, module, sec)
                all_chunks.extend(text_chunks)

    return all_chunks


def process_file(path: Path) -> List[Dict]:
    """Read file, build chunks, and return list of JSON-serializable payloads."""
    if any(skip in str(path) for skip in IGNORE_PATTERNS):
        return []

    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"⚠️  Skipping {path} — read error: {e}")
        return []

    # Build chunks with metadata
    chunks = build_chunks(raw, str(path))
    # Convert to JSON-serializable payloads
    payloads = [c.to_payload() for c in chunks]
    return payloads


def process_all():
    total_files = 0
    total_chunks = 0
    for p in PATH_LIST:
        if not p.exists():
            print(f"⚠️  Path not found: {p} — skipping")
            continue
        for root, _, files in os.walk(p):
            for file in files:
                if not file.lower().endswith(".md"):
                    continue
                fp = Path(root) / file
                file_chunks = process_file(fp)
                if not file_chunks:
                    continue

                out_file = OUTPUT_DIR / (fp.stem + "_chunks.jsonl")
                try:
                    with out_file.open("w", encoding="utf-8") as f:
                        for chunk in file_chunks:
                            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"⚠️  Failed to write {out_file}: {e}")
                    continue

                print(f"✔ Processed: {fp} — chunks: {len(file_chunks)} -> {out_file}")
                total_files += 1
                total_chunks += len(file_chunks)

    print("—" * 40)
    print(f"Finished. Files processed: {total_files}. Total chunks: {total_chunks}")


if __name__ == "__main__":
    print("Starting embedding preprocessor...")
    print("Config:")
    print(pformat({
        "PATH_LIST": [str(p) for p in PATH_LIST],
        "OUTPUT_DIR": str(OUTPUT_DIR),
        "CHUNK_SIZE": CHUNK_SIZE,
        "CHUNK_OVERLAP": CHUNK_OVERLAP,
    }))
    process_all()
