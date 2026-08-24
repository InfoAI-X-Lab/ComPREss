import os
import sys
import gzip
import shutil
import re
from typing import Any, Dict, Optional

# ==================== Configuration ====================

# Mapping table file path (.ass.gz)
MAP_FILE = ""

# Source folder
INPUT_DIR = "clean_gcf"

# Final output folder
OUTPUT_DIR = "final_gcf"


# ================================================

def sanitize_name(name):

    # Clean species name to prevent illegal characters in filename
    name = name.replace(" ", "_")
    name = re.sub(r'[\/\\\:\*\?\"\<\>\|]', '_', name)  # Remove slashes and other special symbols
    name = re.sub(r'[^\w\.-]', '', name)
    return name[:100]


def get_total_sequence_length(file_path):

    # Sum all FASTA sequence lines and ignore headers/blank lines.
    total_length = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith('>'):
                    continue
                total_length += len(line)
    except Exception as e:
        print(f"[Error] Failed to calculate sequence length for {file_path}: {e}")
        return -1

    return total_length


def select_and_rename_files(
    map_file: Optional[str] = None,
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    strict: bool = False
) -> Dict[str, Any]:
    target_map_file = map_file if map_file is not None else MAP_FILE
    target_input_dir = input_dir if input_dir is not None else INPUT_DIR
    target_output_dir = output_dir if output_dir is not None else OUTPUT_DIR

    if not os.path.exists(target_input_dir):
        message = f"Error: Source folder does not exist -> {target_input_dir}"
        if strict:
            raise FileNotFoundError(message)
        print(message)
        return {
            "scanned_files": 0,
            "species_count": 0,
            "copied_count": 0,
            "output_dir": target_output_dir,
        }

    if not os.path.exists(target_map_file):
        message = f"Error: Mapping file does not exist -> {target_map_file}"
        if strict:
            raise FileNotFoundError(message)
        print(message)
        return {
            "scanned_files": 0,
            "species_count": 0,
            "copied_count": 0,
            "output_dir": target_output_dir,
        }

    os.makedirs(target_output_dir, exist_ok=True)

    # --- Load mapping table ---
    print("--- Step 1/3: Loading mapping table (extracting GCF and species names) ---")

    gcf_species_map = {}

    try:
        if target_map_file.endswith(".gz"):
            f_map = gzip.open(target_map_file, 'rt', encoding='utf-8')
        else:
            f_map = open(target_map_file, 'r', encoding='utf-8')

        with f_map as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue

                parts = line.split('\t')

                # Column 3 (idx 2) is GCF, Column 5 (idx 4) is species name
                if len(parts) >= 5:
                    gcf_id = parts[2].strip()
                    raw_species_name = parts[4].strip()

                    if gcf_id not in gcf_species_map:
                        clean_species = sanitize_name(raw_species_name)
                        gcf_species_map[gcf_id] = clean_species

    except Exception as e:
        if strict:
            raise RuntimeError(f"Failed to read mapping table: {e}") from e
        print(f"Failed to read mapping table: {e}")
        return {
            "scanned_files": 0,
            "species_count": 0,
            "copied_count": 0,
            "output_dir": target_output_dir,
        }

    print(f"Mapping table loaded.")

    # --- Scan and group ---
    print(f"--- Step 2/3: Scanning files in {target_input_dir} and grouping by species ---")

    species_files_dict = {}
    scan_count = 0

    files = [f for f in os.listdir(target_input_dir) if f.endswith(".fna")]

    for filename in files:
        file_path = os.path.join(target_input_dir, filename)
        gcf_id = filename.replace(".fna", "")

        if gcf_id in gcf_species_map:
            species = gcf_species_map[gcf_id]
            total_length = get_total_sequence_length(file_path)

            if total_length < 0:
                scan_count += 1
                continue

            if species not in species_files_dict:
                species_files_dict[species] = []

            species_files_dict[species].append({
                'path': file_path,
                'gcf': gcf_id,
                'total_length': total_length
            })

        scan_count += 1
        if scan_count % 50000 == 0:
            print(f"  Scanned {scan_count} files...")

    total_species = len(species_files_dict)
    print(f"Scan complete! {total_species} distinct species identified.")

    # --- Select best and copy ---
    print("--- Step 3/3: Starting organization (select max total length per GCF file -> rename -> copy) ---")

    copied_count = 0

    # Iterate through each species for processing
    for idx, (species, file_list) in enumerate(species_files_dict.items()):

        # Display: progress + species name
        print(f"[{idx + 1}/{total_species}] Detected {species}, organizing...")

        # Sort and get the largest
        sorted_files = sorted(file_list, key=lambda x: x['total_length'], reverse=True)
        winner = sorted_files[0]

        new_filename = f"{species}_{winner['gcf']}.fna"

        src_path = winner['path']
        dst_path = os.path.join(target_output_dir, new_filename)

        try:
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        except IOError as e:
            print(f"  [Error] Copy failed: {new_filename} -> {e}")

    print("-" * 50)
    print("[Task Completed]")
    # === Final statistics ===
    print(f"Statistics:")
    print(f"1. Original files scanned: {scan_count}")
    print(f"2. Total species organized: {total_species}")
    print(f"3. Total files generated: {copied_count} (in {target_output_dir} folder)")
    print("-" * 50)

    return {
        "scanned_files": scan_count,
        "species_count": total_species,
        "copied_count": copied_count,
        "output_dir": target_output_dir,
    }


if __name__ == "__main__":
    select_and_rename_files()
