import json
from pathlib import Path
# List your files
files = [Path("./docs/processed/lab_valid_symbols.json"), Path("./docs/processed/lab2_valid_symbols.json"), Path("./docs/processed/sim_valid_symbols.json")]

merged = {}

for filename in files:
    with open(filename, "r") as f:
        data = json.load(f)
        for key, values in data.items():
            if key not in merged:
                merged[key] = set(values)  # use set to dedupe
            else:
                merged[key].update(values)  # merge and dedupe

# Convert sets back to sorted lists
merged = {k: sorted(list(v)) for k, v in merged.items()}

# Save merged file
with open("docs/processed/merged_valid_symbols.json", "w") as f:
    json.dump(merged, f, indent=2)

print("Merged JSON saved as merged_valid_symbols.json")
