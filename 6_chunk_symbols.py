import json
from pathlib import Path

# Paths
INPUT_JSON = Path("docs/processed/isaacsim_valid_symbols.json")
OUTPUT_DIR = Path("docs/processed/symbol_chunks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Chunk settings
MAX_CHARS = 8192   # roughly ~2000 tokens per chunk

# Load valid symbols
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

chunk_text = ""
chunk_index = 1

def save_chunk(text, index):
    file_path = OUTPUT_DIR / f"valid_symbols_chunk{index}.md"
    file_path.write_text(text.strip(), encoding="utf-8")
    print(f"✔ Saved {file_path}")

for module, symbols in data.items():
    section = [f"# {module}", "Valid symbols in this module:"]
    section += [f"- `{sym}`" for sym in symbols]
    section_text = "\n".join(section) + "\n\n"

    if len(chunk_text) + len(section_text) > MAX_CHARS:
        save_chunk(chunk_text, chunk_index)
        chunk_index += 1
        chunk_text = ""

    chunk_text += section_text

# Save any remaining text
if chunk_text.strip():
    save_chunk(chunk_text, chunk_index)

print(f"✅ Finished splitting {len(data)} modules into {chunk_index} chunks.")
