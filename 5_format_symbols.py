import json
from pathlib import Path

# Input JSON of valid symbols
INPUT_FILE = Path("docs/processed/isaacsim_valid_symbols.json")

# Output markdown file for embedding
OUTPUT_FILE = Path("docs/processed/valid_symbols.md")

# Load the JSON
with INPUT_FILE.open("r", encoding="utf-8") as f:
    symbols = json.load(f)

# Generate markdown content
lines = [
    "# Valid Symbols in Isaac Sim 5.0",
    "",
    "The following symbols are valid and should be used in generated code:",
    ""
]

for sym in symbols:
    lines.append(f"- `{sym}`")

# Save to markdown
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"✔ Saved symbol doc with {len(symbols)} entries to {OUTPUT_FILE}")