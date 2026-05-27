import os
from Bio import SeqIO
import hashlib
from typing import Any, Dict, Optional

# ================= Configuration =================

# 1. Folder path to be processed
INPUT_DIR = ""


# ===========================================

def get_file_content_fingerprint(file_path):

    sequences = []
    try:
        # Read all sequences from FASTA file
        for record in SeqIO.parse(file_path, "fasta"):
            seq_str = str(record.seq).strip().upper()
            if seq_str:
                sequences.append(seq_str)

        if not sequences:
            return None

        sequences.sort()

        combined_seq = "".join(sequences)

        return hashlib.md5(combined_seq.encode('utf-8')).hexdigest()

    except Exception as e:
        print(f"[Error] Failed to read file {file_path}: {e}")
        return None


def remove_strict_duplicates(input_dir: Optional[str] = None, strict: bool = False) -> Dict[str, Any]:
    target_dir = input_dir if input_dir is not None else INPUT_DIR

    if not os.path.exists(target_dir):
        message = f"Error: Directory not found {target_dir}"
        if strict:
            raise FileNotFoundError(message)
        print(message)
        return {
            "total_files": 0,
            "deleted_count": 0,
            "empty_error_count": 0,
            "kept_count": 0,
            "processed_dir": target_dir,
        }

    # Get all .fna files and sort by filename to ensure reproducibility
    files = sorted([f for f in os.listdir(target_dir) if f.endswith(".fna")])
    total_files = len(files)

    if total_files == 0:
        print("Directory is empty or contains no .fna files.")
        return {
            "total_files": 0,
            "deleted_count": 0,
            "empty_error_count": 0,
            "kept_count": 0,
            "processed_dir": target_dir,
        }

    print(f"Starting strict scan of {total_files} files...")

    # Record existing fingerprints
    seen_hashes = set()

    deleted_count = 0
    kept_count = 0
    empty_error_count = 0

    for idx, filename in enumerate(files):
        file_path = os.path.join(target_dir, filename)

        # Calculate fingerprint
        file_hash = get_file_content_fingerprint(file_path)

        if file_hash is None:
            # File is empty or format error, skip and count
            empty_error_count += 1
            continue

        if file_hash in seen_hashes:
            try:
                os.remove(file_path)
                deleted_count += 1
            except OSError as e:
                print(f"[Error] Failed to delete {filename}: {e}")
        else:
            seen_hashes.add(file_hash)
            kept_count += 1

        # Progress bar
        if (idx + 1) % 5000 == 0:
            print(f"Progress: {idx + 1}/{total_files} | Kept: {kept_count} | Deleted: {deleted_count}")

    print("-" * 30)
    print("Deduplication complete")
    print(f"1. Total files scanned: {total_files}")
    print(f"2. Duplicate files deleted: {deleted_count}")
    print(f"3. Abnormal/empty files: {empty_error_count} (not processed)")
    print(f"4. Files kept: {kept_count}")
    print(f"Processed directory: {target_dir}")

    return {
        "total_files": total_files,
        "deleted_count": deleted_count,
        "empty_error_count": empty_error_count,
        "kept_count": kept_count,
        "processed_dir": target_dir,
    }


if __name__ == "__main__":
    remove_strict_duplicates()
