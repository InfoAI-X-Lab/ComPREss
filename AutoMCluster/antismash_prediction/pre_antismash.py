import os
from os import path, listdir
import subprocess
from multiprocessing import Pool
import logging
import json
import pandas as pd
import time


class AntismashRunner:
    def __init__(self):
        # ================= Configuration =================

        # 1. Input: Organized .fna folder
        self.genome_path = ""

        # 2. Output: antiSMASH results folder
        self.antismash_output_path = ""

        # 3. Summary result filename
        self.summary_file = "antismash_summary.tsv"

        # 4. Number of parallel workers (recommended 10-12)
        self.workers = 10

        # 5. Progress refresh frequency (show progress every N files processed)
        self.log_interval = 50

        # =======================================================

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
        self.logger = logging.getLogger('Antismash')

    # ------------------- Function 1: Batch prediction (with progress bar) -------------------
    def run_prediction(self):
        print(f"{'#' * 10} Step 1: Start Batch AntiSMASH Prediction {'#' * 10}\n")

        if not path.exists(self.antismash_output_path):
            os.makedirs(self.antismash_output_path)

        # Get all .fna files
        all_files = [f for f in listdir(self.genome_path) if f.endswith(".fna")]
        total_files = len(all_files)
        self.logger.info(f"Found {total_files} genomes to process.")

        start_time = time.time()

        # Use imap_unordered instead of map to enable real-time counting
        with Pool(self.workers) as p:

            for i, _ in enumerate(p.imap_unordered(self.run_single_task, all_files), 1):

                if i % self.log_interval == 0 or i == total_files:
                    elapsed_time = time.time() - start_time
                    percent = (i / total_files) * 100

                    avg_time_per_file = elapsed_time / i
                    remaining_files = total_files - i
                    if avg_time_per_file > 0:
                        eta_seconds = remaining_files * avg_time_per_file
                        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                    else:
                        eta_str = "Calculating..."

                    print(
                        f"--> [Progress] {i}/{total_files} ({percent:.2f}%) | Elapsed: {int(elapsed_time)}s | ETA: {eta_str}")

    def run_single_task(self, fna_filename):

        base_name = fna_filename.replace(".fna", "")
        output_dir = path.join(self.antismash_output_path, base_name)
        input_file = path.join(self.genome_path, fna_filename)

        # Resume from breakpoint check
        if path.exists(output_dir) and path.exists(path.join(output_dir, "index.html")):
            return base_name  # Return filename for counting

        cmd = [
            "antismash",
            "--taxon", "bacteria",
            "--genefinding-tool", "prodigal",
            "--cpus", "1",
            "--output-dir", output_dir,
            "--output-basename", base_name,
            input_file
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:

            pass

        return base_name

    # ------------------- Function 2: Result statistics -------------------
    def run_gene_count(self):
        print(f"\n{'#' * 10} Step 2: Start Generating Summary Table {'#' * 10}")
        print("Note: This may take a few minutes to scan all JSON files...")

        if not path.exists(self.antismash_output_path):
            self.logger.error("Result directory does not exist!")
            return

        res_dirs = listdir(self.antismash_output_path)
        data_list = []
        count = 0
        total = len(res_dirs)

        for folder_name in res_dirs:
            # Find corresponding JSON file
            json_path = path.join(self.antismash_output_path, folder_name, f"{folder_name}.json")

            if not path.exists(json_path):
                continue

            try:
                with open(json_path, 'r') as f:
                    record_data = json.load(f)

                obj = {}
                # Traverse JSON structure
                for record in record_data.get('records', []):
                    for feature in record.get('features', []):
                        if feature.get('type') == 'region':
                            products = feature.get('qualifiers', {}).get('product', [])
                            for p in products:
                                if p in obj:
                                    obj[p] += 1
                                else:
                                    obj[p] = 1

                if obj:
                    df_item = pd.DataFrame.from_dict(obj, orient='index', columns=[folder_name])
                    data_list.append(df_item)

            except Exception as e:
                self.logger.warning(f"Error parsing JSON for {folder_name}: {e}")

            count += 1
            if count % 1000 == 0:
                print(f"  Processed {count}/{total} folders...")

        # Merge all data
        print("Merging data... (This might take memory)")
        if data_list:
            final_df = pd.concat(data_list, axis=1).fillna(0).T
            final_df.to_csv(self.summary_file, sep="\t")
            print(f"{'-' * 30}\nSuccess! Summary saved to: {self.summary_file}")
            print(f"Shape: {final_df.shape} (Rows=Genomes, Cols=ClusterTypes)")
        else:
            print("No valid data found to summarize.")


if __name__ == '__main__':
    runner = AntismashRunner()

    # Mode 1: Run prediction
    runner.run_prediction()

    # Mode 2: Run statistics
    # runner.run_gene_count()
