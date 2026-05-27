import os
import sys
import gzip
import glob
from typing import Any, Dict, Optional

# ==================== Configuration ====================

# Mapping table file path (.ass.gz)
MAP_FILE = ""

# Script will scan all .fna files in this folder
INPUT_DIR = ""

# All organized GCF files will be placed here
OUTPUT_DIR = "clean_gcf"


# =============================================================

def distribute_sequences(
    map_file: Optional[str] = None,
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    strict: bool = False
) -> Dict[str, Any]:
    target_map_file = map_file if map_file is not None else MAP_FILE
    target_input_dir = input_dir if input_dir is not None else INPUT_DIR
    target_output_dir = output_dir if output_dir is not None else OUTPUT_DIR

    if not os.path.exists(target_input_dir):
        message = f"Error: Input folder does not exist -> {target_input_dir}"
        if strict:
            raise FileNotFoundError(message)
        print(message)
        print("Please open the script and modify the INPUT_DIR variable to the correct path!")
        return {
            "source_files": 0,
            "processed_headers": 0,
            "mapping_count": 0,
            "output_dir": target_output_dir,
        }

    if not os.path.exists(target_map_file):
        message = f"Error: Mapping file does not exist -> {target_map_file}"
        if strict:
            raise FileNotFoundError(message)
        print(message)
        return {
            "source_files": 0,
            "processed_headers": 0,
            "mapping_count": 0,
            "output_dir": target_output_dir,
        }

    if not os.path.exists(target_output_dir):
        os.makedirs(target_output_dir)
        print(f"Output directory created: {target_output_dir}")
    else:
        print(f"Output directory already exists: {target_output_dir}")

    # --- Load mapping table (build NZ -> GCF mapping) ---
    print(f"--- Step 1/3: Loading mapping table into memory (large memory usage, please wait) ---")

    # Mapping dictionary: Key=NZ ID, Value=GCF ID
    nz_to_gcf_map = {}

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

                if len(parts) >= 3:
                    nz_id = parts[0].strip()
                    gcf_id = parts[2].strip()
                    nz_to_gcf_map[nz_id] = gcf_id

    except Exception as e:
        if strict:
            raise RuntimeError(f"Failed to read mapping table: {e}") from e
        print(f"Failed to read mapping table: {e}")
        return {
            "source_files": 0,
            "processed_headers": 0,
            "mapping_count": 0,
            "output_dir": target_output_dir,
        }

    print(f"Mapping loaded! {len(nz_to_gcf_map)} sequence relationships recorded in memory.")

    # Get all .fna file list
    input_files = sorted(glob.glob(os.path.join(target_input_dir, "*.fna")))
    total_files = len(input_files)

    print(f"--- Step 2/3: Scanning {total_files} source files and distributing ---")

    if total_files == 0:
        print("Warning: No .fna files found in input folder, please check path configuration!")
        return {
            "source_files": 0,
            "processed_headers": 0,
            "mapping_count": len(nz_to_gcf_map),
            "output_dir": target_output_dir,
        }

    # Statistics counter
    processed_lines = 0

    for idx, fna_file in enumerate(input_files):
        print(f"[{idx + 1}/{total_files}] Processing: {os.path.basename(fna_file)} ...")

        try:
            with open(fna_file, 'r') as f:
                target_gcf = None

                for line in f:

                    if line.startswith('>'):
                        processed_lines += 1
                        header_parts = line.strip().split()
                        full_nz_id = header_parts[0][1:]  # Remove '>'

                        if full_nz_id in nz_to_gcf_map:
                            target_gcf = nz_to_gcf_map[full_nz_id]
                        else:
                            base_id = full_nz_id.split('.')[0]
                            target_gcf = nz_to_gcf_map.get(base_id)

                        if not target_gcf:
                            continue

                        output_path = os.path.join(target_output_dir, f"{target_gcf}.fna")
                        with open(output_path, 'a') as out_f:
                            out_f.write(line)

                    else:
                        if target_gcf:
                            output_path = os.path.join(target_output_dir, f"{target_gcf}.fna")
                            with open(output_path, 'a') as out_f:
                                out_f.write(line)

        except Exception as e:
            if strict:
                raise RuntimeError(f"Error processing file {fna_file}: {e}") from e
            print(f"Error processing file {fna_file}: {e}")

    print("-" * 50)
    print(f"Task completed! All GCF files saved in: {os.path.abspath(target_output_dir)}")
    return {
        "source_files": total_files,
        "processed_headers": processed_lines,
        "mapping_count": len(nz_to_gcf_map),
        "output_dir": target_output_dir,
    }


if __name__ == "__main__":
    distribute_sequences()
