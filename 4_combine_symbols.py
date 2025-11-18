import re
import json
from pathlib import Path

# Folder containing your cleaned chunked docs
CHUNKS_DIR = Path("docs/processed/chunks")
OUTPUT_FILE = Path("docs/processed/isaacsim_valid_symbols.json")

valid_symbols = set()

# Allowed prefixes
PREFIXES = ("isaaclab.", "isaacsim.", "omni.")

# Scan all markdown files for classes, functions, and imports
for file in CHUNKS_DIR.rglob("*.md"):
    text = file.read_text(errors="ignore")
    
    # Extract documented classes
    for cls in re.findall(r'\bclass (\w+)\b', text):
        # Prepend dummy module if necessary? Only keep full paths with allowed prefixes
        if any(cls.startswith(prefix) for prefix in PREFIXES):
            valid_symbols.add(cls)
    
    # Extract documented functions
    for fn in re.findall(r'\bdef (\w+)\b', text):
        if any(fn.startswith(prefix) for prefix in PREFIXES):
            valid_symbols.add(fn)
    
    # Extract full documented imports
    for imp in re.findall(r'from ([\w\.]+) import (\w+)', text):
        full_import = f"{imp[0]}.{imp[1]}"
        if any(full_import.startswith(prefix) for prefix in PREFIXES):
            valid_symbols.add(full_import)

# Organize by module
module_dict = {}
for symbol in valid_symbols:
    parts = symbol.split(".")
    if len(parts) < 2:
        continue
    module = ".".join(parts[:-1])
    name = parts[-1]
    module_dict.setdefault(module, []).append(name)

# Sort each module's symbols
for mod in module_dict:
    module_dict[mod].sort()

# Save to JSON
with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    json.dump(module_dict, f, indent=2)

print(f"✔ Saved {len(module_dict)} modules with symbols to {OUTPUT_FILE}")
