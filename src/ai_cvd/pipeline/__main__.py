"""Automated raw-telemetry processing, verification and fictional evaluation."""

import argparse
import json
from pathlib import Path

from .fixture import generate_raw
from .runner import run_pipeline, save_json, verify


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser(
        "demo", help="Raw fictional CSV -> DuckDB/Parquet -> model evaluation"
    )
    demo.add_argument("--output", type=Path, default=Path("outputs/pipeline"))
    demo.add_argument("--subjects", type=int, default=12)
    demo.add_argument("--seed", type=int, default=17)
    run = commands.add_parser("run", help="Process source-neutral local exports; no model fitting")
    run.add_argument("--inputs", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--engine", choices=("reference", "vector"), default="reference")
    run.add_argument("--threads", type=int, default=2)
    run.add_argument("--memory-limit", default="512MB")
    check = commands.add_parser("verify", help="Verify completed run and every generated artifact")
    check.add_argument("run", type=Path)
    bench = commands.add_parser(
        "benchmark", help="Synthetic processing benchmark; no predictive experiments"
    )
    bench.add_argument("--output", type=Path, default=Path("outputs/benchmark"))
    bench.add_argument("--subjects", type=int, nargs="+", default=[2, 4])
    bench.add_argument("--days", type=int, default=2)
    bench.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args(argv)
    if args.command == "demo":
        from ai_cvd.demo import run_demo

        from .study_b import model_fixture

        if args.subjects < 6:
            parser.error("The model demo needs at least six fictional subjects")
        args.output.mkdir(parents=True, exist_ok=False)
        generate_raw(args.output / "raw", subjects=args.subjects, seed=args.seed)
        record = run_pipeline(args.output / "raw", args.output / "processed")
        run_demo(
            args.seed, args.output / "evaluation", fixture=model_fixture(args.output / "processed")
        )
        print(
            f"Fictional raw-to-evaluation pipeline complete: {record['raw_rows']} raw rows, {record['subjects']} subjects."
        )
        print("No clinical results were regenerated. Outputs are local generated artifacts.")
    elif args.command == "run":
        record = run_pipeline(
            args.inputs,
            args.output,
            resume=args.resume,
            engine=args.engine,
            threads=args.threads,
            memory_limit=args.memory_limit,
        )
        print(f"Processing complete: {record['raw_rows']} raw rows; {record['subjects']} subjects.")
    elif args.command == "verify":
        record = verify(args.run)
        print(f"Verified {len(record['artifacts'])} generated artifacts.")
    else:
        if args.repeats < 1 or args.days < 1 or any(n < 1 for n in args.subjects):
            parser.error("Benchmark sizes, days and repeats must be positive")
        args.output.mkdir(parents=True, exist_ok=False)
        result = []
        for size in args.subjects:
            raw = args.output / f"raw-{size}"
            generate_raw(raw, subjects=size, days=args.days, immediate=True)
            for engine in ("reference", "vector"):
                for repetition in range(args.repeats):
                    destination = args.output / f"run-{size}-{engine}-{repetition}"
                    run_pipeline(raw, destination, engine=engine)
                    result.append(
                        {
                            "repetition": repetition,
                            **json.loads((destination / "performance.json").read_text()),
                        }
                    )
        save_json(args.output / "benchmark.json", result)
        print(f"Completed {len(result)} synthetic processing benchmarks; no models fitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
