"""Single 'nohup-queue' command with subcommands."""

import argparse
import os
import sys


def _sub_run(args) -> int:
    from .runner import launch_job
    launch_job(
        args.script,
        pids=args.pids,
        num_gpus=args.num_gpus,
        log_dir=args.log_dir,
    )
    return 0


def _sub_jobs(args) -> int:
    from .display import display_jobs, display_gpu_status
    display_jobs(show_all=args.all)
    display_gpu_status()
    return 0


def _sub_clean(args) -> int:
    from .display import cmd_clean, cmd_clear_done
    cmd_clean()
    if args.clear_done:
        cmd_clear_done()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="nohup-queue",
        description="GPU job queue with nohup-style background execution.",
    )
    subs = parser.add_subparsers(dest="command", metavar="COMMAND")
    subs.required = True

    # --- run ---
    default_gpus = int(os.environ.get("NUM_GPUS", "0"))
    p_run = subs.add_parser("run", help="Queue a bash script as a background job")
    p_run.add_argument("script", help="Bash script to run")
    p_run.add_argument(
        "--pids", default="",
        help="Comma-separated PIDs to wait for before starting",
    )
    p_run.add_argument(
        "--num-gpus", type=int, default=default_gpus, dest="num_gpus", metavar="N",
        help="GPUs to claim before running (default: $NUM_GPUS or 0)",
    )
    p_run.add_argument(
        "--log-dir", default=None, dest="log_dir", metavar="DIR",
        help="Log directory (default: ./logs/<task_name>/)",
    )
    p_run.set_defaults(func=_sub_run)

    # --- jobs ---
    p_jobs = subs.add_parser("jobs", help="Display job queue and GPU status")
    p_jobs.add_argument("-a", "--all", action="store_true", help="Show all history including done jobs")
    p_jobs.set_defaults(func=_sub_jobs)

    # --- clean ---
    p_clean = subs.add_parser("clean", help="Remove stale claims and dead job records")
    p_clean.add_argument(
        "--clear-done", action="store_true", dest="clear_done",
        help="Also remove done/failed records older than 24h",
    )
    p_clean.set_defaults(func=_sub_clean)

    args = parser.parse_args()
    return args.func(args)
