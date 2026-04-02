"""
Extract and display TensorBoard metrics in compact tabular format.

Samples epochs linearly (1, 10, 20, 30, 40, last) for trend visibility.
Two-phase workflow: essential metrics first, then drill-down on demand.

Usage:
    python -m modules.tools.extract_metrics run_003
    python -m modules.tools.extract_metrics run_003 --metrics Loss/train,Loss/test
    python -m modules.tools.extract_metrics run_003 --list
    python -m modules.tools.extract_metrics run_003 --all
"""

import argparse
import os
import sys

import config

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Default essential metrics (Phase 1)
DEFAULT_METRICS = [
    "Loss/train",
    "Loss/test",
    "Accuracy/train",
    "Accuracy/test",
    "Metrics/final_accuracy",
    "Metrics/final_accuracy_count",
    "Metrics/f1_macro",
    "Metrics/gap_loss",
    "Metrics/gap_accuracy",
    "Metrics/entropy_mean",
    "Metrics/highconf_05_bear_acc",
    "Metrics/highconf_05_bull_acc",
    "Metrics/highconf_05_count",
    "Metrics/highconf_06_bear_acc",
    "Metrics/highconf_06_bull_acc",
    "Metrics/highconf_06_count",
    "Metrics/highconf_07_bear_acc",
    "Metrics/highconf_07_bull_acc",
    "Metrics/highconf_07_count",
    "Metrics/confmargin_top05_acc",
    "Metrics/confmargin_top05_count",
    "Metrics/confmargin_top10_acc",
    "Metrics/confmargin_top10_count",
    "Metrics/confmargin_top25_acc",
    "Metrics/confmargin_top25_count",
    "Metrics/confmargin_top50_acc",
    "Metrics/confmargin_top50_count",
]

# Default sample points (epochs)
DEFAULT_SAMPLE_POINTS = [1, 10, 20, 30, 40]


def load_accumulator(run_dir: str) -> EventAccumulator:
    """Load TensorBoard events from run directory."""
    acc = EventAccumulator(run_dir)
    acc.Reload()
    return acc


def get_available_metrics(acc: EventAccumulator) -> list:
    """Return sorted list of all scalar tags in this run."""
    return sorted(acc.Tags().get("scalars", []))


def get_metric_values(acc: EventAccumulator, tag: str) -> list:
    """Return list of (step, value) for a given metric tag."""
    try:
        events = acc.Scalars(tag)
        return [(e.step, e.value) for e in events]
    except KeyError:
        return []


def sample_epochs(values: list, sample_points: list) -> dict:
    """
    Sample metric values at specified epoch points.
    Returns {epoch: value} for each sample point found.
    Adds the last epoch automatically.
    """
    if not values:
        return {}

    # Build step->value mapping
    step_map = {step: val for step, val in values}
    all_steps = sorted(step_map.keys())

    if not all_steps:
        return {}

    last_step = all_steps[-1]

    # Collect sample points + last epoch
    targets = sorted(set(sample_points + [last_step]))

    result = {}
    for target in targets:
        if target in step_map:
            result[target] = step_map[target]
        else:
            # Find closest epoch
            closest = min(all_steps, key=lambda s: abs(s - target))
            # Only use if within reasonable range (±3 epochs)
            if abs(closest - target) <= 3:
                result[target] = step_map[closest]

    return result


def format_table(run_name: str, metrics_data: dict, total_epochs: int) -> str:
    """
    Format metrics as compact table.

    Args:
        run_name: Name of the run
        metrics_data: {metric_tag: {epoch: value, ...}, ...}
        total_epochs: Total number of epochs in run
    """
    if not metrics_data:
        return f"Run: {run_name} - No data found"

    # Collect all sampled epochs across metrics
    all_epochs = sorted(set(
        ep for samples in metrics_data.values() for ep in samples.keys()
    ))

    if not all_epochs:
        return f"Run: {run_name} - No sampled epochs found"

    # Shorten metric names for display
    short_names = {}
    for tag in metrics_data:
        # Remove common prefixes for compactness
        short = tag.replace("Metrics/", "").replace("Train/", "tr_")
        short_names[tag] = short

    # Build header
    col_width = max(10, max(len(s) for s in short_names.values()) + 1)
    header = f"{'Epoch':>6}"
    for tag in metrics_data:
        header += f"  {short_names[tag]:>{col_width}}"

    # Build separator
    sep = "-" * len(header)

    # Build rows
    rows = []
    for ep in all_epochs:
        row = f"{ep:>6}"
        for tag in metrics_data:
            val = metrics_data[tag].get(ep)
            if val is not None:
                if abs(val) < 0.001:
                    row += f"  {val:>{col_width}.5f}"
                elif abs(val) > 100:
                    row += f"  {val:>{col_width}.1f}"
                else:
                    row += f"  {val:>{col_width}.4f}"
            else:
                row += f"  {'---':>{col_width}}"
        rows.append(row)

    # Assemble
    lines = [
        f"Run: {run_name} ({total_epochs} epochs)",
        sep,
        header,
        sep,
    ]
    lines.extend(rows)
    lines.append(sep)

    return "\n".join(lines)


def list_metrics(acc: EventAccumulator, run_name: str) -> str:
    """List all available metrics with their epoch count."""
    tags = get_available_metrics(acc)
    if not tags:
        return f"Run: {run_name} - No metrics found"

    lines = [f"Run: {run_name} - Available metrics ({len(tags)}):"]
    for tag in tags:
        values = get_metric_values(acc, tag)
        n = len(values)
        last_val = values[-1][1] if values else None
        val_str = f"  last={last_val:.4f}" if last_val is not None else ""
        lines.append(f"  {tag:<45} ({n} pts){val_str}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Extract TensorBoard metrics")
    parser.add_argument("run", help="Run name (e.g. run_003) or full path")
    parser.add_argument("--metrics", "-m", type=str, default=None,
                        help="Comma-separated metric tags to extract")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List all available metrics")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Show all available metrics (not just defaults)")
    parser.add_argument("--epochs", "-e", type=str, default=None,
                        help="Comma-separated epoch sample points (default: 1,10,20,30,40,last)")

    args = parser.parse_args()

    # Resolve run directory
    if os.path.isdir(args.run):
        run_dir = args.run
        run_name = os.path.basename(run_dir)
    else:
        run_dir = os.path.join(config.TENSORBOARD_DIR, args.run)
        run_name = args.run

    if not os.path.isdir(run_dir):
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)

    # Load events
    acc = load_accumulator(run_dir)

    # List mode
    if args.list:
        print(list_metrics(acc, run_name))
        return

    # Determine which metrics to extract
    if args.metrics:
        metric_tags = [m.strip() for m in args.metrics.split(",")]
    elif args.all:
        metric_tags = get_available_metrics(acc)
    else:
        metric_tags = DEFAULT_METRICS

    # Determine sample points
    if args.epochs:
        sample_points = [int(e.strip()) for e in args.epochs.split(",")]
    else:
        sample_points = DEFAULT_SAMPLE_POINTS

    # Extract and sample
    metrics_data = {}
    for tag in metric_tags:
        values = get_metric_values(acc, tag)
        if values:
            sampled = sample_epochs(values, sample_points)
            if sampled:
                metrics_data[tag] = sampled

    # Determine total epochs
    total_epochs = 0
    for tag in metric_tags:
        values = get_metric_values(acc, tag)
        if values:
            total_epochs = max(total_epochs, max(s for s, _ in values))

    # Display
    print(format_table(run_name, metrics_data, total_epochs))

    # Show missing metrics
    missing = [t for t in metric_tags if t not in metrics_data]
    if missing:
        print(f"\nMissing: {', '.join(missing)}")


if __name__ == "__main__":
    main()
