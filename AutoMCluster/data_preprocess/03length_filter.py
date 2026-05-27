import os
import gzip
from typing import Any, Dict

# ================= Configuration =================

# 1. Input folder
INPUT_FOLDER = 'clean_fna'

# 2. Output folder
OUTPUT_DIR = 'clean_900_fna'

# 3. Length threshold
THRESHOLD = 900


# =============================================================

def filter_sequences_by_length(
    input_folder: str,
    output_folder: str,
    min_length: int = 900,
    strict: bool = False
) -> Dict[str, Any]:

    # Iterate through folder, remove sequences with length less than min_length, and count total

    # Create output directory
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Output directory created: {output_folder}")
    else:
        print(f"Output directory already exists: {output_folder}")

    # Get all supported files
    if not os.path.exists(input_folder):
        message = f"Error: Input folder not found '{input_folder}'"
        if strict:
            raise FileNotFoundError(message)
        print(message)
        return {
            "files_processed": 0,
            "original_sequences": 0,
            "kept_sequences": 0,
            "removed_sequences": 0,
        }

    # Support .fna, .fasta, .fa and their .gz compressed files
    files = [f for f in os.listdir(input_folder) if f.endswith(('.fna', '.fna.gz', '.fasta', '.fasta.gz', '.fa'))]
    total_files = len(files)

    if total_files == 0:
        print(f"No sequence files found in '{input_folder}'")
        return {
            "files_processed": 0,
            "original_sequences": 0,
            "kept_sequences": 0,
            "removed_sequences": 0,
        }

    print(f"Starting to process {total_files} files, length threshold: {min_length} bp\n")
    print(f"{'Filename':<40} | {'Original':<10} | {'Kept(>={min_length})':<15} | {'Filtered(<{min_length})':<15}")
    print("-" * 95)

    # === Global statistics variables ===
    grand_total_original = 0
    grand_kept = 0
    grand_removed = 0

    for file_name in files:
        input_path = os.path.join(input_folder, file_name)

        # Auto-generate output filename (remove .gz suffix uniformly)
        if file_name.endswith('.gz'):
            output_name = file_name.replace('.gz', '')
            open_func = gzip.open
            mode = 'rt'
        else:
            output_name = file_name
            open_func = open
            mode = 'r'

        output_path = os.path.join(output_folder, output_name)

        try:
            local_total = 0
            local_kept = 0

            with open_func(input_path, mode, encoding='utf-8') as f_in, \
                    open(output_path, 'w', encoding='utf-8') as f_out:

                header = None
                sequence_parts = []

                for line in f_in:
                    line = line.strip()
                    if not line: continue

                    if line.startswith('>'):
                        if header:
                            local_total += 1
                            seq_str = "".join(sequence_parts)
                            # Core judgment: whether length >= 900
                            if len(seq_str) >= min_length:
                                f_out.write(f"{header}\n")
                                f_out.write(f"{seq_str}\n")
                                local_kept += 1

                        header = line
                        sequence_parts = []
                    else:
                        sequence_parts.append(line)

                if header:
                    local_total += 1
                    seq_str = "".join(sequence_parts)
                    if len(seq_str) >= min_length:
                        f_out.write(f"{header}\n")
                        f_out.write(f"{seq_str}\n")
                        local_kept += 1

            # Calculate removed count for current file
            local_removed = local_total - local_kept

            # Accumulate to global variables
            grand_total_original += local_total
            grand_kept += local_kept
            grand_removed += local_removed

            print(f"{file_name:<40} | {local_total:<10} | {local_kept:<15} | {local_removed:<15}")

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    # === Final statistics report ===
    print("-" * 95)
    print(f"All done! Statistics:")
    print(f"Original sequences total: {grand_total_original}")
    print(f"Kept sequences total    : {grand_kept}")
    print(f"Removed sequences total : {grand_removed}")

    if grand_total_original > 0:
        loss_rate = (grand_removed / grand_total_original) * 100
        print(f"Overall removal rate    : {loss_rate:.2f}%")

    print(f"Results saved in        : ./{output_folder}/")

    return {
        "files_processed": total_files,
        "original_sequences": grand_total_original,
        "kept_sequences": grand_kept,
        "removed_sequences": grand_removed,
    }


if __name__ == "__main__":
    filter_sequences_by_length(INPUT_FOLDER, OUTPUT_DIR, THRESHOLD)
