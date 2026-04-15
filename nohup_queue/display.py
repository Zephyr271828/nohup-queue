"""Display and maintenance commands for njobs."""

import os
import subprocess
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Set, Tuple

from .gpu import load_claims, live_gpu_processes, pid_is_alive, clean_stale_claims
from .state import get_jobs_dir, load_jobs, update_job_status


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def live_job_pids(jobs: Dict[str, Dict]) -> Set[int]:
    """Return the set of live PIDs among all job records."""
    pids: Set[int] = set()
    for job in jobs.values():
        try:
            pid = int(job.get("pid"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if pid_is_alive(pid):
            pids.add(pid)
    return pids


@lru_cache(maxsize=128)
def host_status(hostname: str) -> str:
    if not hostname:
        return "--"
    if hostname == os.uname().nodename:
        return "local"
    try:
        r = subprocess.run(
            ["getent", "hosts", hostname],
            capture_output=True, text=True, timeout=2,
        )
        return "resolves" if r.returncode == 0 and r.stdout.strip() else "missing"
    except Exception:
        return "unknown"


@lru_cache(maxsize=256)
def slurm_status(slurm_job_id: str) -> str:
    if not slurm_job_id:
        return "--"
    try:
        r = subprocess.run(
            ["squeue", "-h", "-j", slurm_job_id, "-o", "%T"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0:
            s = r.stdout.strip()
            return s if s else "gone"
        return "unknown"
    except Exception:
        return "unknown"


def format_time(iso_string: str) -> str:
    if not iso_string:
        return "--"
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
        delta = datetime.utcnow() - dt.replace(tzinfo=None)
        secs = delta.total_seconds()
        if secs < 3600:
            return f"{int(secs / 60)}m ago"
        elif secs < 86400:
            return f"{int(secs / 3600)}h ago"
        else:
            return f"{int(secs / 86400)}d ago"
    except Exception:
        return iso_string[:10]


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_jobs(show_all: bool = False) -> None:
    jobs = load_jobs()
    claims = load_claims()
    gpu_procs = live_gpu_processes()
    live_pids = live_job_pids(jobs)

    if not jobs:
        print("No jobs found.")
        return

    if not show_all:
        jobs = {k: v for k, v in jobs.items() if v.get("status") != "done"}

    sorted_jobs = sorted(
        jobs.items(),
        key=lambda x: x[1].get("queued_at", ""),
        reverse=True,
    )

    print(
        f"{'JOB ID':<12} {'TASK':<25} {'STATUS':<12} {'NODE':<20} "
        f"{'NODE OK':<9} {'SLURM':<10} {'GPUs':<12} {'QUEUED':<12} {'LOG':<40}"
    )
    print("-" * 157)

    for job_id, job in sorted_jobs:
        task_name = Path(job.get("script_path", "unknown")).stem
        status = job.get("status", "unknown")
        queued_at = job.get("queued_at", "")
        log_file = job.get("log_file", "")

        job_pid = job.get("pid")
        job_pid_live = isinstance(job_pid, int) and job_pid in live_pids
        node_name = job.get("hostname") or (os.uname().nodename if job_pid_live else "--")
        node_ok = host_status(node_name)
        slurm_state = slurm_status(str(job.get("slurm_job_id", "")))

        gpu_list = sorted(idx for idx, (jid, _) in claims.items() if jid == job_id)
        if not gpu_list and job_pid_live and isinstance(job_pid, int):
            gpu_list = sorted(
                idx for idx, pids in gpu_procs.items() if job_pid in pids
            )
        if not gpu_list:
            try:
                gpu_list = sorted(
                    int(p.strip())
                    for p in job.get("gpus_claimed", "").split(",")
                    if p.strip()
                )
            except ValueError:
                gpu_list = []

        gpu_str = ",".join(str(g) for g in gpu_list) if gpu_list else "--"
        log_name = Path(log_file).name if log_file else "--"

        print(
            f"{job_id:<12} {task_name:<25} {status:<12} {node_name:<20} "
            f"{node_ok:<9} {slurm_state:<10} {gpu_str:<12} "
            f"{format_time(queued_at):<12} {log_name:<40}"
        )

    print()


def display_gpu_status() -> None:
    claims = load_claims()
    jobs = load_jobs()
    gpu_procs = live_gpu_processes()
    live_pids = live_job_pids(jobs)

    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return

        gpus_info = {}
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                try:
                    idx = int(parts[0])
                    gpus_info[idx] = (parts[1], float(parts[2]), float(parts[3]))
                except ValueError:
                    pass

        if not gpus_info:
            return

        print("GPU STATUS:")
        for gpu_idx in sorted(gpus_info.keys()):
            name, used, total = gpus_info[gpu_idx]
            if gpu_idx in claims:
                job_id, _ = claims[gpu_idx]
                job = jobs.get(job_id, {})
                task_name = Path(job.get("script_path", "unknown")).stem
                claimed_at = job.get("started_at", "")
                print(f"  GPU {gpu_idx}: CLAIMED by {job_id} ({task_name}) since {format_time(claimed_at)}")
            elif gpu_procs.get(gpu_idx):
                used_pct = (used / total * 100) if total > 0 else 0
                matching = [
                    jid for jid, job in jobs.items()
                    if isinstance(job.get("pid"), int)
                    and job["pid"] in live_pids
                    and job["pid"] in gpu_procs[gpu_idx]
                ]
                owner = ",".join(sorted(matching)) if matching else "external process"
                print(f"  GPU {gpu_idx}: IN USE by {owner} ({used:.0f}/{total:.0f} MiB, {used_pct:.1f}%)")
            else:
                used_pct = (used / total * 100) if total > 0 else 0
                print(f"  GPU {gpu_idx}: FREE ({used:.0f}/{total:.0f} MiB, {used_pct:.1f}%)")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Maintenance commands
# ---------------------------------------------------------------------------

def cmd_clean() -> None:
    """Remove stale GPU claims and dead running job records."""
    clean_stale_claims()

    cleaned = 0
    jobs = load_jobs()
    current_host = os.uname().nodename
    jobs_dir = get_jobs_dir()
    for job_id, job in jobs.items():
        if job.get("status") != "running":
            continue
        if job.get("hostname") != current_host:
            continue
        try:
            pid = int(job.get("pid"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pid = -1
        if pid_is_alive(pid):
            continue
        job_file = jobs_dir / f"{job_id}.json"
        try:
            job_file.unlink()
            cleaned += 1
        except FileNotFoundError:
            pass

    print(f"Cleaned stale claims and removed {cleaned} dead job records.")


def cmd_clear_done() -> None:
    """Remove done/failed job records older than 24 h."""
    cutoff = datetime.utcnow() - timedelta(hours=24)
    jobs = load_jobs()
    jobs_dir = get_jobs_dir()
    count = 0
    for job_id, job in jobs.items():
        if job.get("status") not in ("done", "failed"):
            continue
        ended_at = job.get("ended_at")
        if not ended_at:
            continue
        try:
            ended = datetime.fromisoformat(ended_at.replace("Z", "+00:00")).replace(tzinfo=None)
            if ended < cutoff:
                job_file = jobs_dir / f"{job_id}.json"
                job_file.unlink()
                count += 1
        except Exception:
            pass
    print(f"Cleared {count} old job records.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Display job queue status.")
    parser.add_argument("-a", "--all", action="store_true", help="Show all job history")
    parser.add_argument("--clean", action="store_true", help="Clean stale claims and dead jobs")
    parser.add_argument("--clear-done", action="store_true", help="Clear done/failed jobs older than 24h")
    args = parser.parse_args()

    if args.clean:
        cmd_clean()
        return 0
    if args.clear_done:
        cmd_clear_done()
        return 0

    display_jobs(show_all=args.all)
    display_gpu_status()
    return 0
