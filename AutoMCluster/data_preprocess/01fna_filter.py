import os
import gzip
import pandas as pd
from typing import Any, Dict, Set



# ================= Configuration =================

BLACKLIST_CSV = ''
INPUT_FOLDER = '.'
OUTPUT_DIR_NAME = 'clean_fna'


# =============================================================

def load_blacklist(csv_path: str, strict: bool = False) -> Set[str]:
    print(f"Loading blacklist: {csv_path} ...")
    try:
        df = pd.read_csv(csv_path, header=None)
        blacklist = set(df[0].astype(str).str.strip())
        print(f"Blacklist loaded, total {len(blacklist)} entries.")
        return blacklist
    except Exception as e:
        if strict:
            raise RuntimeError(f"Failed to read CSV: {e}") from e
        print(f"Failed to read CSV: {e}")
        return set()


def process_and_summarize(
    input_folder: str,
    blacklist: Set[str],
    output_dir: str,
    strict: bool = False
) -> Dict[str, Any]:
    # 1. Create output directory
    if not os.path.exists(input_folder):
        message = f"Input folder not found: {input_folder}"
        if strict:
            raise FileNotFoundError(message)
        print(message)
        return {
            "files_processed": 0,
            "original_sequences": 0,
            "kept_sequences": 0,
            "removed_sequences": 0,
            "error_files": 0,
        }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all .fna.gz files
    gz_files = [f for f in os.listdir(input_folder) if f.endswith('.fna.gz')]
    total_files = len(gz_files)

    if total_files == 0:
        print("Warning: No .fna.gz files found in the current folder!")
        return {
            "files_processed": 0,
            "original_sequences": 0,
            "kept_sequences": 0,
            "removed_sequences": 0,
            "error_files": 0,
        }

    print(f"\nFound {total_files} compressed files, starting processing...\n")
    print(f"{'Filename':<40} | {'Original':<10} | {'Remaining':<10} | {'Removed':<8}")
    print("-" * 85)

    # === Initialize global counters ===
    grand_total_original = 0
    grand_total_kept = 0
    grand_total_removed = 0
    error_files = 0

    for file_name in gz_files:
        input_path = os.path.join(input_folder, file_name)
        output_name = file_name.replace('.gz', '')
        output_path = os.path.join(output_dir, output_name)

        try:
            local_total = 0
            local_kept = 0

            with gzip.open(input_path, 'rt', encoding='utf-8') as f_in, \
                    open(output_path, 'w', encoding='utf-8') as f_out:

                header = None
                sequence_parts = []
                should_keep = True

                for line in f_in:
                    line = line.strip()
                    if not line: continue

                    if line.startswith(">"):
                        if header:
                            local_total += 1
                            if should_keep:
                                f_out.write(f"{header}\n")
                                f_out.write(f"{''.join(sequence_parts)}\n")
                                local_kept += 1

                        header = line
                        sequence_parts = []

                        should_keep = True
                        for bad_id in blacklist:
                            if bad_id in header:
                                should_keep = False
                                break
                    else:
                        sequence_parts.append(line)

                if header:
                    local_total += 1
                    if should_keep:
                        f_out.write(f"{header}\n")
                        f_out.write(f"{''.join(sequence_parts)}\n")
                        local_kept += 1

            local_removed = local_total - local_kept

            print(f"{file_name:<40} | {local_total:<10} | {local_kept:<10} | {local_removed:<8}")

            grand_total_original += local_total
            grand_total_kept += local_kept
            grand_total_removed += local_removed

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
            error_files += 1

    # === Print summary report ===
    print("-" * 85)
    print("\n" + "=" * 30 + " Processing Summary Report " + "=" * 30)
    print(f" Files processed    : {total_files}")
    print(f" Original sequences : {grand_total_original}")
    print(f" Remaining sequences: {grand_total_kept}")
    print(f" Removed sequences  : {grand_total_removed}")

    if grand_total_original > 0:
        percent = (grand_total_removed / grand_total_original) * 100
        print(f" Removal rate       : {percent:.2f}%")

    print("=" * 76 + "\n")
    print(f"Results saved in ./{output_dir}/ folder")
    return {
        "files_processed": total_files,
        "original_sequences": grand_total_original,
        "kept_sequences": grand_total_kept,
        "removed_sequences": grand_total_removed,
        "error_files": error_files,
    }


def run_fna_filter(
    blacklist_csv: str,
    input_folder: str,
    output_dir: str,
    strict: bool = False
) -> Dict[str, Any]:
    if not os.path.exists(blacklist_csv):
        message = f"Blacklist file not found: {blacklist_csv}"
        if strict:
            raise FileNotFoundError(message)
        print(message)
        return {
            "files_processed": 0,
            "original_sequences": 0,
            "kept_sequences": 0,
            "removed_sequences": 0,
            "error_files": 0,
        }

    bad_ids = load_blacklist(blacklist_csv, strict=strict)
    return process_and_summarize(input_folder, bad_ids, output_dir, strict=strict)



if __name__ == "__main__":
    run_fna_filter(BLACKLIST_CSV, INPUT_FOLDER, OUTPUT_DIR_NAME, strict=False)
