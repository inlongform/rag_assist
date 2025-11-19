import json
from pathlib import Path

merged_file = Path("./docs/processed/merged_valid_symbols.json")
jsonl_file = Path("./docs/processed/chunks/merged_valid_symbols.jsonl")

with open(merged_file, "r", encoding="utf-8") as f:
    merged = json.load(f)

# Convert into a list of documents compatible with ingestion
with open(jsonl_file, "w", encoding="utf-8") as f:
    for key, items in merged.items():
        content = "\n".join(items)  # combine array items into a single string
        chunk = {
            "id": key,
            "content": content
        }
        f.write(json.dumps(chunk) + "\n")

print(f"Saved {len(merged)} chunks to {jsonl_file}")
