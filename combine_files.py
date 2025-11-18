import shutil
from pathlib import Path

SOURCE_DIR = Path("./docs/examples_md")
DEST_DIR = Path("./examples_combined")

def flatten_examples():
    DEST_DIR.mkdir(exist_ok=True)

    for md_file in SOURCE_DIR.rglob("*.md"):
        # Preserve uniqueness by prefixing with relative path
        rel_path = md_file.relative_to(SOURCE_DIR)
        # Replace folder separators with underscores
        flat_name = "_".join(rel_path.parts)
        out_path = DEST_DIR / flat_name

        # Copy file into the combined directory
        shutil.copy2(md_file, out_path)
        print(f"Moved {md_file} -> {out_path}")

if __name__ == "__main__":
    flatten_examples()
    print(f"✅ All Markdown files moved into {DEST_DIR}")
