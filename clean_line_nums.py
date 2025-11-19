import os
import re

def strip_line_numbers(text: str) -> str:
    cleaned = []
    for line in text.splitlines(keepends=True):
        new_line = re.sub(r'^\s*\d+\s*', '', line)
        cleaned.append(new_line.lstrip())
    return "".join(cleaned)


def process_directory(directory: str, output_directory: str):
    os.makedirs(output_directory, exist_ok=True)

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".md"):
                input_path = os.path.join(root, file)

                # Mirror directory structure in output folder
                relative = os.path.relpath(input_path, directory)
                output_path = os.path.join(output_directory, relative)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                with open(input_path, "r", encoding="utf-8") as f:
                    original = f.read()

                cleaned = strip_line_numbers(original)

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(cleaned)

                print(f"Created: {output_path}")


if __name__ == "__main__":
    INPUT_DIR = "./docs/lab_docs"
    OUTPUT_DIR = "./docs/lab_docs_cleaned"

    process_directory(INPUT_DIR, OUTPUT_DIR)
    print("Done, created cleaned copies.")
