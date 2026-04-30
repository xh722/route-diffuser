"""Build an experiment registry and leaderboard from report artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from planner.reports import build_experiment_registry, build_leaderboard_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="JSON report artifact paths to index")
    parser.add_argument("--output-dir", default="outputs/registry")
    parser.add_argument("--primary-metric", default="fde")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = build_experiment_registry(
        args.paths,
        metadata={
            "primary_metric": args.primary_metric,
            "num_inputs": len(args.paths),
        },
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = output_dir / "experiment_registry.json"
    leaderboard_path = output_dir / "leaderboard.md"
    registry.save_json(registry_path)
    leaderboard_path.write_text(
        build_leaderboard_markdown(registry, primary_metric=args.primary_metric),
        encoding="utf-8",
    )

    print(f"saved registry to {registry_path}")
    print(f"saved leaderboard to {leaderboard_path}")


if __name__ == "__main__":
    main()
