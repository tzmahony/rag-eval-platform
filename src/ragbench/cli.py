"""Command-line interface: ``ragbench run|report|calibrate|models``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _cmd_run(args: argparse.Namespace) -> int:
    from ragbench.experiment import ExperimentConfig, run_experiment

    cfg = ExperimentConfig.from_file(args.config)
    if args.limit is not None:
        cfg.limit = args.limit
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)
    if args.no_cache:
        cfg.cache = False
    if args.concurrency:
        cfg.concurrency = args.concurrency
    result = run_experiment(cfg, quiet=args.quiet)
    if not args.quiet:
        print((result.output_dir / "report.md").read_text())
    print(f"\nReport: {result.output_dir / 'report.md'}\nHTML:   {result.output_dir / 'report.html'}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from ragbench.experiment import load_run
    from ragbench.report import write_reports

    records, summary = load_run(args.run_dir)
    write_reports(records, summary, Path(args.run_dir))
    print((Path(args.run_dir) / "report.md").read_text())
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    import yaml

    from ragbench.calibration import calibrate, format_calibration
    from ragbench.judges import make_judge

    judge_cfg = yaml.safe_load(args.judge) if args.judge else {"type": "heuristic"}
    if isinstance(judge_cfg, str):
        judge_cfg = {"type": "llm", "model": judge_cfg} if ":" in judge_cfg else {"type": judge_cfg}
    report = calibrate(make_judge(judge_cfg), args.labels)
    print(format_calibration(report))
    return 0


def _cmd_models(_: argparse.Namespace) -> int:
    from ragbench.llm.pricing import DEFAULT_PRICES

    print(f"{'model id':34} {'$ / 1M in':>10} {'$ / 1M out':>11}")
    for mid, p in DEFAULT_PRICES.items():
        print(f"{mid:34} {p.input_per_mtok:>10.2f} {p.output_per_mtok:>11.2f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragbench", description="Evaluate RAG configurations and LLMs.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run an experiment from a YAML config")
    run.add_argument("config")
    run.add_argument("--limit", type=int, help="Evaluate only the first N examples")
    run.add_argument("--output-dir", help="Override output_dir")
    run.add_argument("--concurrency", type=int)
    run.add_argument("--no-cache", action="store_true", help="Disable the LLM response cache")
    run.add_argument("-q", "--quiet", action="store_true")
    run.set_defaults(func=_cmd_run)

    rep = sub.add_parser("report", help="Re-render reports for an existing run directory")
    rep.add_argument("run_dir")
    rep.set_defaults(func=_cmd_report)

    cal = sub.add_parser("calibrate", help="Measure judge agreement with human labels")
    cal.add_argument("labels", help="JSONL of human-labelled examples")
    cal.add_argument("--judge", help="'heuristic', a model id such as 'anthropic:claude-haiku-5-5', or YAML")
    cal.set_defaults(func=_cmd_calibrate)

    mod = sub.add_parser("models", help="List built-in model pricing")
    mod.set_defaults(func=_cmd_models)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
