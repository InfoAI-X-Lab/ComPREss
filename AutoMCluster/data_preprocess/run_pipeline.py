import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


class ConfigError(Exception):
    pass


class StepExecutionError(Exception):
    pass


STEP_SPECS: Dict[str, Dict[str, Any]] = {
    "fna_filter": {
        "script": "01fna_filter.py",
        "function": "run_fna_filter",
        "description": "Read .fna.gz files and a blacklist CSV, then write filtered clean_fna output.",
        "input_key": "input_folder",
        "output_key": "output_dir",
        "required_keys": ["enabled", "blacklist_csv", "input_folder", "output_dir"],
    },
    "deduplicate": {
        "script": "deduplicate_fna.py",
        "function": "remove_strict_duplicates",
        "description": "Perform strict deduplication on .fna files based on sequence-content fingerprints.",
        "input_key": "input_dir",
        "output_key": "input_dir",
        "required_keys": ["enabled", "input_dir"],
    },
    "stat_length": {
        "script": "02stat_length.py",
        "function": "analyze_and_recommend",
        "description": "Analyze .fna/.fna.gz sequence length distribution and print threshold recommendations.",
        "input_key": "input_folder",
        "output_key": None,
        "required_keys": ["enabled", "input_folder"],
    },
    "length_filter": {
        "script": "03length_filter.py",
        "function": "filter_sequences_by_length",
        "description": "Prompt for a length threshold at runtime, then filter short sequences and write FASTA files.",
        "input_key": "input_folder",
        "output_key": "output_dir",
        "required_keys": ["enabled", "input_folder", "output_dir"],
    },
    "nz_to_gcf": {
        "script": "04nz_to_gcf.py",
        "function": "distribute_sequences",
        "description": "Use .ass/.ass.gz mapping data to group NZ sequences into GCF files.",
        "input_key": "input_dir",
        "output_key": "output_dir",
        "required_keys": ["enabled", "map_file", "input_dir", "output_dir"],
    },
    "final_gcf": {
        "script": "05final_gcf.py",
        "function": "select_and_rename_files",
        "description": "For species with multiple GCF files, keep the one with the largest total sequence length.",
        "input_key": "input_dir",
        "output_key": "output_dir",
        "required_keys": ["enabled", "map_file", "input_dir", "output_dir"],
    },
    "count_fna": {
        "script": "count_fna_quantity.py",
        "function": "batch_count",
        "description": "Count FASTA sequence entries in the target folder or file.",
        "input_key": "target_path",
        "output_key": None,
        "required_keys": ["enabled", "target_path"],
    },
}


STEP_PATH_REQUIREMENTS: Dict[str, List[Tuple[str, str]]] = {
    "fna_filter": [("blacklist_csv", "file"), ("input_folder", "dir")],
    "deduplicate": [("input_dir", "dir")],
    "stat_length": [("input_folder", "dir")],
    "length_filter": [("input_folder", "dir")],
    "nz_to_gcf": [("map_file", "file"), ("input_dir", "dir")],
    "final_gcf": [("map_file", "file"), ("input_dir", "dir")],
    "count_fna": [("target_path", "exists")],
}


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path


def load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise StepExecutionError(f"Unable to load script: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in config file: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Unable to read config file: {exc}") from exc

    required_top = ["pipeline", "execution_order", "steps"]
    missing_top = [key for key in required_top if key not in data]
    if missing_top:
        raise ConfigError(f"Missing top-level config fields: {', '.join(missing_top)}")

    pipeline_cfg = data["pipeline"]
    if not isinstance(pipeline_cfg, dict):
        raise ConfigError("Config field 'pipeline' must be an object")

    for key in ["show_step_summary", "stop_on_error"]:
        if key not in pipeline_cfg:
            raise ConfigError(f"Missing pipeline field: {key}")
        if not isinstance(pipeline_cfg[key], bool):
            raise ConfigError(f"pipeline.{key} must be a boolean")

    execution_order = data["execution_order"]
    if not isinstance(execution_order, list) or not execution_order:
        raise ConfigError("execution_order must be a non-empty array")

    steps_cfg = data["steps"]
    if not isinstance(steps_cfg, dict):
        raise ConfigError("steps must be an object")

    for step_id in execution_order:
        if step_id not in STEP_SPECS:
            raise ConfigError(f"Unknown step in execution_order: {step_id}")
        if step_id not in steps_cfg:
            raise ConfigError(f"Missing step config in steps: {step_id}")
        if not isinstance(steps_cfg[step_id], dict):
            raise ConfigError(f"steps.{step_id} must be an object")

        required_step_keys = STEP_SPECS[step_id]["required_keys"]
        missing_step = [key for key in required_step_keys if key not in steps_cfg[step_id]]
        if missing_step:
            raise ConfigError(f"steps.{step_id} is missing fields: {', '.join(missing_step)}")

        if not isinstance(steps_cfg[step_id].get("enabled"), bool):
            raise ConfigError(f"steps.{step_id}.enabled must be a boolean")

    return data


def validate_step_paths(step_id: str, step_cfg: Dict[str, Any], base_dir: Path) -> None:
    for field, kind in STEP_PATH_REQUIREMENTS.get(step_id, []):
        raw_value = step_cfg[field]
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise StepExecutionError(f"[{step_id}] Field '{field}' must be a non-empty string")
        path = resolve_path(base_dir, raw_value)
        if kind == "file" and not path.is_file():
            raise StepExecutionError(f"[{step_id}] File does not exist: {path}")
        if kind == "dir" and not path.is_dir():
            raise StepExecutionError(f"[{step_id}] Input directory does not exist: {path}")
        if kind == "exists" and not path.exists():
            raise StepExecutionError(f"[{step_id}] Path does not exist: {path}")


def ask_yes_no(prompt: str) -> bool:
    while True:
        answer = input(prompt).strip().lower()
        if answer in {"yes", "y"}:
            return True
        if answer in {"no", "n"}:
            return False
        print("Invalid input. Please enter yes or no.")


def ask_positive_integer(prompt: str) -> int:
    while True:
        answer = input(prompt).strip()
        try:
            value = int(answer)
        except ValueError:
            print("Invalid input. Please enter a positive integer.")
            continue

        if value > 0:
            return value
        print("Invalid input. Please enter a positive integer.")


def print_step_info(index: int, total: int, step_id: str, step_cfg: Dict[str, Any]) -> None:
    spec = STEP_SPECS[step_id]
    input_value = step_cfg.get(spec["input_key"], "N/A") if spec["input_key"] else "N/A"
    output_value = step_cfg.get(spec["output_key"], "N/A") if spec["output_key"] else "N/A"

    print("\n" + "=" * 80)
    print(f"Step {index}/{total}")
    print(f"Step ID          : {step_id}")
    print(f"Script           : {spec['script']}")
    print(f"Description      : {spec['description']}")
    print(f"Input path       : {input_value}")
    print(f"Output path      : {output_value}")
    print(f"Default enabled  : {step_cfg.get('enabled')}")
    if step_id == "length_filter":
        print("Threshold        : Prompt at runtime after reviewing the previous statistics output")
    print("=" * 80)


def execute_step(
    step_id: str,
    step_cfg: Dict[str, Any],
    base_dir: Path,
    module_cache: Dict[str, Any],
) -> Any:
    spec = STEP_SPECS[step_id]
    script_name = spec["script"]
    script_path = base_dir / script_name

    if script_name not in module_cache:
        module_cache[script_name] = load_module_from_path(
            module_name=f"pipeline_{script_name.replace('.', '_')}",
            module_path=script_path,
        )

    module = module_cache[script_name]
    func = getattr(module, spec["function"], None)
    if func is None:
        raise StepExecutionError(f"Script {script_name} is missing function {spec['function']}")

    if step_id == "fna_filter":
        return func(
            blacklist_csv=step_cfg["blacklist_csv"],
            input_folder=step_cfg["input_folder"],
            output_dir=step_cfg["output_dir"],
            strict=True,
        )
    if step_id == "deduplicate":
        return func(
            input_dir=step_cfg["input_dir"],
            strict=True,
        )
    if step_id == "stat_length":
        return func(
            input_folder=step_cfg["input_folder"],
            strict=True,
        )
    if step_id == "length_filter":
        threshold = ask_positive_integer(
            "Enter the sequence length threshold for this step (positive integer): "
        )
        return func(
            input_folder=step_cfg["input_folder"],
            output_folder=step_cfg["output_dir"],
            min_length=threshold,
            strict=True,
        )
    if step_id == "nz_to_gcf":
        return func(
            map_file=step_cfg["map_file"],
            input_dir=step_cfg["input_dir"],
            output_dir=step_cfg["output_dir"],
            strict=True,
        )
    if step_id == "final_gcf":
        return func(
            map_file=step_cfg["map_file"],
            input_dir=step_cfg["input_dir"],
            output_dir=step_cfg["output_dir"],
            strict=True,
        )
    if step_id == "count_fna":
        return func(
            target_path=step_cfg["target_path"],
            strict=True,
        )

    raise StepExecutionError(f"No executor implemented for step: {step_id}")


def print_summary(success: List[str], skipped: List[str], failed: List[str]) -> None:
    def fmt(items: List[str]) -> str:
        return ", ".join(items) if items else "None"

    print("\n" + "#" * 80)
    print("Pipeline Execution Summary")
    print("#" * 80)
    print(f"Successful steps ({len(success)}): {fmt(success)}")
    print(f"Skipped steps    ({len(skipped)}): {fmt(skipped)}")
    print(f"Failed steps     ({len(failed)}): {fmt(failed)}")
    print("#" * 80)


def run_pipeline(config_path: Path) -> int:
    config = load_config(config_path)
    base_dir = config_path.parent

    pipeline_cfg = config["pipeline"]
    execution_order = config["execution_order"]
    steps_cfg = config["steps"]
    stop_on_error = pipeline_cfg["stop_on_error"]
    show_step_summary = pipeline_cfg["show_step_summary"]

    success_steps: List[str] = []
    skipped_steps: List[str] = []
    failed_steps: List[str] = []
    module_cache: Dict[str, Any] = {}

    total = len(execution_order)
    for idx, step_id in enumerate(execution_order, start=1):
        step_cfg = steps_cfg[step_id]
        print_step_info(idx, total, step_id, step_cfg)

        should_run = ask_yes_no("Run this step? Enter yes/no: ")
        if not should_run:
            skipped_steps.append(step_id)
            print(f"[SKIPPED] {step_id}")
            continue

        try:
            validate_step_paths(step_id, step_cfg, base_dir)
            result = execute_step(step_id, step_cfg, base_dir, module_cache)
            success_steps.append(step_id)
            print(f"[SUCCESS] {step_id}")
            if show_step_summary:
                print(f"[STEP RESULT] {result}")
        except Exception as exc:
            failed_steps.append(step_id)
            print(f"[FAILED] {step_id}: {exc}")
            if stop_on_error:
                remaining = execution_order[idx:]
                skipped_steps.extend(remaining)
                print("stop_on_error=true, so the pipeline will stop here. Remaining steps are marked as skipped.")
                break
            print("stop_on_error=false, continuing with the remaining steps.")

    print_summary(success_steps, skipped_steps, failed_steps)
    return 0 if not failed_steps else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive controller for the data preprocessing pipeline")
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to the config file (default: config.json in the same directory as this script)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline_dir = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = pipeline_dir / config_path

    try:
        raise SystemExit(run_pipeline(config_path))
    except ConfigError as exc:
        print(f"[CONFIG ERROR] {exc}")
        raise SystemExit(2)
