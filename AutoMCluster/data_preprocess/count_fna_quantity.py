import os
from typing import Any, Dict



# ================= Configuration =================

# Enter the filename to count, or a folder path containing multiple files
# TARGET_PATH = 'all_unique_sequences.fna'  <-- Count a single file
# TARGET_PATH = '.'                         <-- Count all files in current folder
TARGET_PATH = 'final_gcf'


# =============================================================

def count_sequences_in_file(file_path):

    # Count the number of sequences in a single FASTA file
    # Principle: Count lines starting with '>'

    count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith('>'):
                    count += 1
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return 0

    return count


def batch_count(target_path: str, strict: bool = False) -> Dict[str, Any]:

    total_all = 0

    # If target is a folder, count all .fna files in the folder
    if os.path.isdir(target_path):
        print(f"Counting all .fna files in folder '{target_path}'...\n")
        print(f"{'Filename':<40} | {'Sequences':<10}")
        print("-" * 55)

        fna_files = [f for f in os.listdir(target_path) if f.endswith('.fna')]

        if not fna_files:
            print("No .fna files found in this folder.")
            return {
                "target_path": target_path,
                "is_directory": True,
                "file_count": 0,
                "total_sequences": 0,
            }

        for file_name in fna_files:
            full_path = os.path.join(target_path, file_name)
            seq_count = count_sequences_in_file(full_path)

            print(f"{file_name:<40} | {seq_count:<10}")
            total_all += seq_count

        print("-" * 55)
        print(f"Total sequences in all files: {total_all}")
        return {
            "target_path": target_path,
            "is_directory": True,
            "file_count": len(fna_files),
            "total_sequences": total_all,
        }

    # If target is a specific file
    elif os.path.isfile(target_path):
        seq_count = count_sequences_in_file(target_path)
        print(f"File: {target_path}")
        print(f"Sequences: {seq_count}")
        return {
            "target_path": target_path,
            "is_directory": False,
            "file_count": 1,
            "total_sequences": seq_count,
        }

    else:
        message = f"Error: Path not found '{target_path}'"
        if strict:
            raise FileNotFoundError(message)
        print(message)
        return {
            "target_path": target_path,
            "is_directory": False,
            "file_count": 0,
            "total_sequences": 0,
        }



if __name__ == "__main__":
    batch_count(TARGET_PATH)
