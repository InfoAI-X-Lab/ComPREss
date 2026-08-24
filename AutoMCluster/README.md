# antiSMASH Genome Workflow

## Overview
This repository has two stages:

1. `data_preprocess`: clean raw `.fna.gz` data, deduplicate, analyze sequence length, filter by a user-provided threshold, map NZ sequences to GCF, and select final genomes.
2. `antismash_prediction`: run antiSMASH in batch on processed genomes and summarize BGC results.

Recommended preprocessing entry point:

- `data_preprocess/run_pipeline.py`
- `data_preprocess/config.json`

## Project Layout
```text
antismash/
├─ data_preprocess/
│  ├─ run_pipeline.py
│  ├─ config.json
│  ├─ 01fna_filter.py
│  ├─ deduplicate_fna.py
│  ├─ 02stat_length.py
│  ├─ 03length_filter.py
│  ├─ 04nz_to_gcf.py
│  ├─ 05final_gcf.py
│  └─ count_fna_quantity.py
└─ antismash_prediction/
   └─ pre_antismash.py
```

## Preprocessing Pipeline
Run:

```bash
cd data_preprocess
python run_pipeline.py --config config.json
```

The controller prompts interactively for each step in this order:

1. `fna_filter` -> `01fna_filter.py`
2. `deduplicate` -> `deduplicate_fna.py`
3. `stat_length` -> `02stat_length.py`
4. `length_filter` -> `03length_filter.py`
5. `nz_to_gcf` -> `04nz_to_gcf.py`
6. `final_gcf` -> `05final_gcf.py`
7. `count_fna` -> `count_fna_quantity.py`

Why deduplication is after step 01:
`deduplicate_fna.py` processes `.fna` files, while the raw input is usually `.fna.gz`.

For steps 02 and 03:
run `02stat_length.py` first, review the printed statistics, then enter the threshold manually when `03length_filter.py` starts. The threshold is no longer stored in `config.json`.

Prompt rules:

- `yes` / `y`: run the step
- `no` / `n`: skip the step
- invalid input: keep asking until valid input is received

## config.json
The configuration file contains three top-level sections:

- `pipeline`
- `execution_order`
- `steps`

Important options:

- `pipeline.show_step_summary`: print a result summary after each step
- `pipeline.stop_on_error`
  - `true`: stop immediately when a step fails
  - `false`: record the failure and continue

Each step has an `enabled` field shown as the default state. Actual execution is still decided by the interactive prompt.
The `length_filter` step now reads the threshold from runtime input, so `config.json` no longer needs a `threshold` field.

## Preprocessing Scripts
- `01fna_filter.py`: filter `.fna.gz` files using a blacklist CSV and output `.fna`
- `deduplicate_fna.py`: remove duplicate `.fna` files by sequence-content MD5
- `02stat_length.py`: analyze `.fna/.fna.gz` length distribution and print statistics for threshold selection
- `03length_filter.py`: prompt for a threshold at runtime, then filter short sequences
- `04nz_to_gcf.py`: use `.ass/.ass.gz` mapping to group NZ sequences into GCF files
- `05final_gcf.py`: for species with multiple GCF files, keep the one with the maximum total sequence length
- `count_fna_quantity.py`: count sequence quantity in a target folder or file

## pre_antismash.py
Path: `antismash_prediction/pre_antismash.py`

The main class is `AntismashRunner`. Key settings are defined in `__init__`:

- `self.genome_path`: input `.fna` folder, usually the final preprocessing output
- `self.antismash_output_path`: antiSMASH output folder
- `self.summary_file`: summary result filename, default `antismash_summary.tsv`
- `self.workers`: number of parallel workers, default `10`
- `self.log_interval`: progress print interval, default every `50` files

Two modes are supported:

1. Prediction mode

```python
runner.run_prediction()
# runner.run_gene_count()
```

This runs the `antismash` command in parallel. If `index.html` already exists in a sample output directory, that sample is skipped, so resume behavior is supported.

2. Summary mode

```python
# runner.run_prediction()
runner.run_gene_count()
```

This scans antiSMASH JSON outputs, counts cluster types for each genome, and writes a TSV summary.

Run:

```bash
cd antismash_prediction
python pre_antismash.py
```

## Dependencies
- Python 3.9+ recommended
- `biopython`
- `pandas`
- antiSMASH

Install example:

```bash
pip install biopython pandas
```
