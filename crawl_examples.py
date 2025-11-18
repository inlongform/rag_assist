from pathlib import Path

EXAMPLES_DIR = Path(r"C:\Users\rwill\Documents\projects\omniverse\IsaacLab\standalone_examples")
OUTPUT_DIR = Path("examples_md")

def split_examples():
    OUTPUT_DIR.mkdir(exist_ok=True)
    for py_file in EXAMPLES_DIR.rglob("*.py"):
        rel_path = py_file.relative_to(EXAMPLES_DIR)
        # Mirror the directory structure under OUTPUT_DIR
        out_path = OUTPUT_DIR / rel_path.with_suffix(".md")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as out:
            out.write(f"# {rel_path}\n\n")
            out.write("```python\n")
            out.write(py_file.read_text(encoding="utf-8"))
            out.write("\n```\n")

if __name__ == "__main__":
    split_examples()
    print(f"✅ Wrote Markdown files under {OUTPUT_DIR}")
