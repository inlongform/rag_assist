from pathlib import Path
import shutil

INPUT_DIR = Path("docs/processed/cleaned_chunks")
OUTPUT_DIR = Path("docs/processed/final_chunks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 500

def batch_files():
    files = sorted(INPUT_DIR.rglob("*.md"))  # all cleaned chunk files
    total_files = len(files)
    print(f"Total files: {total_files}")

    batch_num = 1
    for i in range(0, total_files, BATCH_SIZE):
        batch_files = files[i:i+BATCH_SIZE]
        batch_folder = OUTPUT_DIR / f"batch_{batch_num:03d}"
        batch_folder.mkdir(parents=True, exist_ok=True)

        for file in batch_files:
            shutil.copy(file, batch_folder / file.name)

        print(f"✔ Batch {batch_num} → {len(batch_files)} files")
        batch_num += 1

if __name__ == "__main__":
    batch_files()
