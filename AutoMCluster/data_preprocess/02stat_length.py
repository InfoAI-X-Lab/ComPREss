import os
import gzip
import math
from typing import Any, Dict



# ================= Configuration =================

# Enter the folder you want to analyze
INPUT_FOLDER = ''


# =============================================================

def analyze_and_recommend(input_folder: str, strict: bool = False) -> Dict[str, Any]:
    # 1. Check folder
    if not os.path.exists(input_folder):
        message = f"Error: Folder not found '{input_folder}'"
        if strict:
            raise FileNotFoundError(message)
        print(message)
        return {
            "total_sequences": 0,
            "min_length": None,
            "max_length": None,
            "average_length": None,
            "median_length": None,
            "p0_1": None,
            "p1": None,
            "p5": None,
        }

    files = [f for f in os.listdir(input_folder) if f.endswith('.fna') or f.endswith('.fna.gz')]

    if not files:
        print(f"No data files found in '{input_folder}'.")
        return {
            "total_sequences": 0,
            "min_length": None,
            "max_length": None,
            "average_length": None,
            "median_length": None,
            "p0_1": None,
            "p1": None,
            "p5": None,
        }

    print(f"Reading data distribution from {len(files)} files (this may take a while)...")

    all_lengths = []

    # Iterate and read all lengths
    for idx, file_name in enumerate(files):
        file_path = os.path.join(input_folder, file_name)

        if (idx + 1) % 100 == 0:
            print(f"   -> Processed {idx + 1} files...")

        if file_name.endswith('.gz'):
            open_func = gzip.open
            mode = 'rt'
        else:
            open_func = open
            mode = 'r'

        try:
            with open_func(file_path, mode, encoding='utf-8') as f:
                sequence_parts = []
                for line in f:
                    line = line.strip()
                    if not line: continue
                    if line.startswith('>'):
                        if sequence_parts:
                            all_lengths.append(len("".join(sequence_parts)))
                        sequence_parts = []
                    else:
                        sequence_parts.append(line)
                if sequence_parts:
                    all_lengths.append(len("".join(sequence_parts)))

        except Exception as e:
            print(f"Error reading {file_name}: {e}")

    total_seqs = len(all_lengths)
    if total_seqs == 0:
        print("No sequences found.")
        return {
            "total_sequences": 0,
            "min_length": None,
            "max_length": None,
            "average_length": None,
            "median_length": None,
            "p0_1": None,
            "p1": None,
            "p5": None,
        }

    print("Data reading complete, performing statistical analysis...\n")

    # === Core statistics ===
    all_lengths.sort()

    min_len = all_lengths[0]
    max_len = all_lengths[-1]
    avg_len = sum(all_lengths) / total_seqs
    median_len = all_lengths[int(total_seqs * 0.5)]

    p0_1 = all_lengths[int(total_seqs * 0.001)]
    p1 = all_lengths[int(total_seqs * 0.01)]
    p5 = all_lengths[int(total_seqs * 0.05)]

    print("=" * 60)
    print("Data Overview")
    print("=" * 60)
    print(f" Total sequences : {total_seqs}")
    print(f" Min length      : {min_len} bp")
    print(f" Max length      : {max_len} bp")
    print(f" Average length  : {int(avg_len)} bp")
    print(f" Median          : {median_len} bp")
    print("-" * 60)

    print("\nThreshold Recommendations (Sensitivity Analysis)")
    print(f"{'Option':<12} | {'Threshold':<10} | {'Sequences to remove':<20} | {'Data loss rate':<15} | {'Description'}")
    print("-" * 95)

    # Option 1: Lenient (remove very few noise)
    loss_p0_1 = int(total_seqs * 0.001)
    print(f"{'Lenient':<12} | {p0_1:<10} | {loss_p0_1:<20} | {'0.1%':<15} | Remove only the most extreme 0.1% fragments")

    # Option 2: Recommended (remove bottom 1% outliers)
    loss_p1 = int(total_seqs * 0.01)
    print(f"{'Recommended':<12} | {p1:<10} | {loss_p1:<20} | {'1.0%':<15} | Best cost-benefit, remove bottom 1% outliers")

    # Option 3: Strict (remove bottom 5%)
    loss_p5 = int(total_seqs * 0.05)
    print(f"{'Strict':<12} | {p5:<10} | {loss_p5:<20} | {'5.0%':<15} | Ensure high quality, but more data loss")

    print("-" * 95)

    # Final recommendation
    print("\nAnalysis Conclusion:")
    if p1 - min_len > 50:
        print(f"  Significant short sequence tail detected! Shortest sequence is only {min_len} bp, while mainstream sequences are at least {p1} bp.")
        print(f"  Strongly recommended threshold: [{p1}]")
    elif p1 == min_len:
        print(f"  Data is very uniform, no obvious short outliers.")
        print(f"  You can keep the current state (no threshold) or set a safety baseline (e.g., 200).")
    else:
        print(f"  Recommended threshold: [{p1}]. This will remove the 1% potential error sequences.")

    return {
        "total_sequences": total_seqs,
        "min_length": min_len,
        "max_length": max_len,
        "average_length": avg_len,
        "median_length": median_len,
        "p0_1": p0_1,
        "p1": p1,
        "p5": p5,
    }



if __name__ == "__main__":
    analyze_and_recommend(INPUT_FOLDER)
