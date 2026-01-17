#!/usr/bin/env python3
"""
Run all possible configurations for Dynamic Clustering Federated Learning experiments.

This script systematically tests all combinations of parameters defined in the project,
running experiments sequentially and saving results to MLflow.
"""

import itertools
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Generator, Union


# ============================================================================
# Configuration Space - All Possible Values
# ============================================================================

CONFIG_SPACE = {
    # Training configuration
    "num-server-rounds": [10, 15, 20],
    "local-epochs": [3, 5],
    # Dataset and model
    "dataset": ["mnist", "fashion_mnist", "cifar10", "cifar100"],
    "model": ["mlp"],  # Currently only MLP is supported
    # Clustering configuration
    "clustering-mode": ["static", "dynamic", "adaptive"],
    "n-client-clusters": [2, 3, 5],
    "clustering-interval": [2, 3, 5],  # Only relevant for dynamic mode
    # Drift simulation configuration
    "drift-type": ["none", "sudden", "gradual", "recurrent", "incremental"],
    "drift-round": [5, 8, 10],
    "drift-magnitude": [0.3, 0.5, 0.7],
    "drift-threshold": [0.2, 0.3, 0.4],  # Only relevant for adaptive mode
    # Federation configuration (via federation names)
    "federation": ["local-10", "local-20", "local-50"],
}


# ============================================================================
# Configuration Validation & Filtering
# ============================================================================


def is_valid_config(config: Dict[str, Any]) -> bool:
    """Validate configuration based on constraints and dependencies."""

    # clustering-interval only relevant for dynamic mode
    if config["clustering-mode"] != "dynamic":
        # Skip configs with varying clustering-interval when not in dynamic mode
        # We'll normalize this in the config generation
        pass

    # drift-threshold only relevant for adaptive mode
    if config["clustering-mode"] != "adaptive":
        # Skip configs with varying drift-threshold when not in adaptive mode
        pass

    # drift-round must be less than num-server-rounds
    if config["drift-round"] >= config["num-server-rounds"]:
        return False

    # drift-round and drift-magnitude only relevant when drift-type != "none"
    if config["drift-type"] == "none":
        # For no drift, we can ignore drift-round and drift-magnitude variations
        pass

    # num-supernodes (from federation) must be >= n-client-clusters
    num_supernodes = int(config["federation"].split("-")[1])
    if num_supernodes < config["n-client-clusters"]:
        return False

    return True


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize config by removing irrelevant parameters for specific modes."""
    normalized = config.copy()

    # Set default clustering-interval for non-dynamic modes
    if config["clustering-mode"] != "dynamic":
        normalized["clustering-interval"] = 3  # Default value

    # Set default drift-threshold for non-adaptive modes
    if config["clustering-mode"] != "adaptive":
        normalized["drift-threshold"] = 0.3  # Default value

    # Set default drift parameters for no-drift scenarios
    if config["drift-type"] == "none":
        normalized["drift-round"] = 10  # Default value
        normalized["drift-magnitude"] = 0.5  # Default value

    return normalized


def estimate_total_configs() -> int:
    """Estimate total number of possible configurations (before validation/deduplication)."""
    total = 1
    for values in CONFIG_SPACE.values():
        total *= len(values)
    return total


def generate_configs() -> Generator[Dict[str, Any], None, None]:
    """Generate valid configuration combinations using a memory-efficient generator.

    Yields configurations one at a time instead of loading all into memory.
    Automatically filters invalid configs and removes duplicates.
    """
    keys = list(CONFIG_SPACE.keys())
    values = list(CONFIG_SPACE.values())

    # Track seen configs to avoid duplicates (after normalization)
    seen = set()

    # Generate combinations lazily using itertools.product
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))

        # Skip invalid configurations
        if not is_valid_config(config):
            continue

        # Normalize and check for duplicates
        normalized = normalize_config(config)
        config_key = json.dumps(normalized, sort_keys=True)

        if config_key not in seen:
            seen.add(config_key)
            yield normalized


# ============================================================================
# Experiment Execution
# ============================================================================


def build_run_config_string(config: Dict[str, Any]) -> str:
    """Build the --run-config string for flwr run command."""
    # Exclude federation from run-config as it's passed separately
    config_items = []
    for key, value in config.items():
        if key == "federation":
            continue
        if isinstance(value, str):
            config_items.append(f"{key}='{value}'")
        else:
            config_items.append(f"{key}={value}")

    return " ".join(config_items)


def run_experiment(
    config: Dict[str, Any], experiment_id: int, total: Union[int, str], logs_dir: Path
) -> Dict[str, Any]:
    """Run a single experiment with the given configuration.

    Args:
        config: Experiment configuration dictionary
        experiment_id: Unique ID for this experiment
        total: Total number of experiments (int) or '?' if unknown
        logs_dir: Directory to save stdout/stderr log files

    Returns:
        Dictionary with experiment results (excluding full stdout/stderr)
    """

    print(f"\n{'=' * 80}")
    print(f"Experiment {experiment_id}/{total}")
    print(f"{'=' * 80}")
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"{'=' * 80}\n")

    # Build the command
    federation = config["federation"]
    run_config = build_run_config_string(config)

    cmd = ["flwr", "run", ".", federation, "--run-config", run_config]

    # Record start time
    start_time = time.time()
    start_timestamp = datetime.now().isoformat()

    # Run the experiment
    result = {
        "experiment_id": experiment_id,
        "config": config,
        "start_time": start_timestamp,
        "command": " ".join(cmd),
        "status": "unknown",
        "duration_seconds": 0,
        "error": None,
    }

    try:
        print(f"Running command: {' '.join(cmd)}\n")

        # Run the command and capture output
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout per experiment
        )

        # Calculate duration
        duration = time.time() - start_time
        result["duration_seconds"] = duration
        result["end_time"] = datetime.now().isoformat()

        # Write stdout/stderr to log files instead of storing in memory
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_prefix = f"exp_{experiment_id:05d}"
        stdout_file = logs_dir / f"{log_prefix}_stdout.log"
        stderr_file = logs_dir / f"{log_prefix}_stderr.log"

        stdout_file.write_text(process.stdout)
        stderr_file.write_text(process.stderr)

        result["stdout_log"] = str(stdout_file)
        result["stderr_log"] = str(stderr_file)

        # Check if successful
        if process.returncode == 0:
            result["status"] = "success"
            print(
                f"\n✓ Experiment {experiment_id} completed successfully in {duration:.2f}s"
            )
        else:
            result["status"] = "failed"
            # Store error message (typically short) but reference full stderr log
            error_preview = process.stderr[:500] if process.stderr else "Unknown error"
            result["error"] = error_preview
            print(
                f"\n✗ Experiment {experiment_id} failed with return code {process.returncode}"
            )
            print(f"Error preview: {error_preview}")
            print(f"Full stderr log: {stderr_file}")

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        result["duration_seconds"] = duration
        result["end_time"] = datetime.now().isoformat()
        result["status"] = "timeout"
        result["error"] = "Experiment exceeded 1 hour timeout"
        print(f"\n✗ Experiment {experiment_id} timed out after {duration:.2f}s")

    except Exception as e:
        duration = time.time() - start_time
        result["duration_seconds"] = duration
        result["end_time"] = datetime.now().isoformat()
        result["status"] = "error"
        result["error"] = str(e)
        print(f"\n✗ Experiment {experiment_id} encountered an error: {e}")

    return result


# ============================================================================
# Results Persistence
# ============================================================================


def save_results(
    results: List[Dict[str, Any]], output_dir: Path, is_final: bool = False
):
    """Save experiment results to disk.

    Args:
        results: List of experiment results to save
        output_dir: Directory to save results
        is_final: If True, use 'final' prefix; otherwise use 'checkpoint'
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results as JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "final" if is_final else "checkpoint"
    results_file = output_dir / f"{prefix}_experiments_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    if is_final:
        print(f"\n✓ Saved final results to: {results_file}")
    # Only print checkpoint saves occasionally to reduce output

    # Save summary as CSV-style text
    summary_file = output_dir / f"{prefix}_summary_{timestamp}.txt"

    with open(summary_file, "w") as f:
        f.write("Experiment Summary\n")
        f.write("=" * 100 + "\n\n")

        # Statistics
        total = len(results)
        successful = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] == "failed")
        timeout = sum(1 for r in results if r["status"] == "timeout")
        errors = sum(1 for r in results if r["status"] == "error")

        f.write(f"Total experiments: {total}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Timeout: {timeout}\n")
        f.write(f"Errors: {errors}\n\n")

        # Duration statistics
        durations = [r["duration_seconds"] for r in results]
        if durations:
            f.write(
                f"Total duration: {sum(durations):.2f}s ({sum(durations) / 3600:.2f}h)\n"
            )
            f.write(f"Average duration: {sum(durations) / len(durations):.2f}s\n")
            f.write(f"Min duration: {min(durations):.2f}s\n")
            f.write(f"Max duration: {max(durations):.2f}s\n\n")

        # Detailed listing
        f.write("=" * 100 + "\n")
        f.write("Detailed Results\n")
        f.write("=" * 100 + "\n\n")

        for result in results:
            f.write(f"Experiment {result['experiment_id']}\n")
            f.write(f"  Status: {result['status']}\n")
            f.write(f"  Duration: {result['duration_seconds']:.2f}s\n")
            f.write(f"  Config: {json.dumps(result['config'], indent=4)}\n")
            if result["error"]:
                f.write(f"  Error: {result['error']}\n")
            f.write("\n")

    if is_final:
        print(f"✓ Saved summary to: {summary_file}")


# ============================================================================
# Main Execution
# ============================================================================


def main():
    """Main function to run all experiments."""

    print("=" * 80)
    print("Dynamic Clustering FL - Comprehensive Experiment Runner")
    print("=" * 80)
    print()

    # Estimate configuration space size
    print("Analyzing configuration space...")
    estimated_total = estimate_total_configs()
    print(f"Estimated possible combinations: {estimated_total:,}")
    print("\nNote: Using memory-efficient generator approach.")
    print(
        "Actual count (after validation & deduplication) will be determined during execution.\n"
    )

    # Setup directories
    output_dir = Path("results/experiment_runs")
    logs_dir = output_dir / "logs"

    # Run experiments using generator (memory efficient)
    results = []
    start_time = time.time()
    experiment_id = 0
    save_interval = 10  # Save checkpoint every 10 experiments

    # Process configs one at a time from generator
    for config in generate_configs():
        experiment_id += 1

        # Note: We don't know total count ahead of time, so use '?' for display
        result = run_experiment(config, experiment_id, "?", logs_dir)
        results.append(result)

        # Save checkpoint periodically (every save_interval experiments)
        if experiment_id % save_interval == 0:
            save_results(results, output_dir, is_final=False)
            print(f"  [Checkpoint saved: {experiment_id} experiments completed]")

    # Check if any experiments ran
    if experiment_id == 0:
        print("No valid configurations found!")
        return

    # Final summary
    total_duration = time.time() - start_time

    print("\n" + "=" * 80)
    print("All Experiments Completed!")
    print("=" * 80)
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
    print(f"Total duration: {total_duration:.2f}s ({total_duration / 3600:.2f}h)")
    print("=" * 80)

    # Save final results
    save_results(results, output_dir, is_final=True)

    print("\n✓ All results saved to results/experiment_runs/")
    print("✓ Experiment logs saved to results/experiment_runs/logs/")
    print("\n✓ Check MLflow UI for detailed metrics and visualizations:")
    print("  mlflow ui")


if __name__ == "__main__":
    main()
