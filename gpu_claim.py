#!/usr/bin/env python3
"""
Atomic GPU claim/release management for job queuing.

Uses os.mkdir() as atomic primitive (safe on WekaFS/NFS) to claim GPU slots.
Each GPU is represented by a directory gpu_N.lock/ in .gpu_queue/claims/.

Subcommands:
  claim --num-gpus N --job-id ID --pid PID
    -> Claims N GPUs, prints indices "0,1,2,3", exits 0
    -> If not enough, exits 1 with no side effects

  release --job-id ID
    -> Removes all gpu_N.lock/ dirs owned by this job

  update-status --job-id ID --status S [--pid P] [--gpus G] [--exit-code E]
    -> Rewrite job JSON with new status, optional fields

  list
    -> Print JSON of all current GPU claims and job states

  clean
    -> Remove stale claims (dead PID) and expired job records
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

# State directory
STATE_DIR = Path.cwd() / ".gpu_queue"
CLAIMS_DIR = STATE_DIR / "claims"
JOBS_DIR = STATE_DIR / "jobs"


def ensure_state_dirs():
    """Create .gpu_queue and subdirectories if they don't exist."""
    CLAIMS_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)


def get_gpu_list() -> List[Tuple[int, float, bool]]:
    """
    Get list of GPUs with free memory and idle status.

    Returns list of (gpu_idx, free_memory_MiB, is_idle)
    """
    try:
        # Get free memory for each GPU
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []

        gpu_mem = {}
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split(",")
            if len(parts) == 2:
                idx, mem = parts[0].strip(), parts[1].strip()
                try:
                    gpu_mem[int(idx)] = float(mem)
                except ValueError:
                    pass

        # Check which GPUs have compute processes (busy)
        gpus = []
        for idx in sorted(gpu_mem.keys()):
            is_idle = True
            try:
                # Check if there are any compute processes on this GPU
                result = subprocess.run(
                    ["nvidia-smi", "-i", str(idx), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    output = result.stdout.strip()
                    # If there's any PID listed, GPU is busy
                    if output and output != "No running processes found":
                        is_idle = False
            except Exception:
                pass

            gpus.append((idx, gpu_mem[idx], is_idle))

        return gpus
    except Exception as e:
        print(f"Error querying GPUs: {e}", file=sys.stderr)
        return []


def pid_is_alive(pid: int) -> bool:
    """Check if a process is still alive via kill -0."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False


def claim_gpus(num_gpus: int, job_id: str, pid: int) -> Optional[str]:
    """
    Atomically claim num_gpus GPUs.

    Returns comma-separated GPU indices on success, None on failure.
    On success, gpu_N.lock/ directories are created with metadata files.
    On failure, rolls back any partial claims.
    """
    if num_gpus == 0:
        return ""

    ensure_state_dirs()
    clean_stale_claims()

    # Get available GPUs sorted by preference (idle first, then by free mem)
    gpus = get_gpu_list()

    # Filter out already-claimed GPUs
    available_gpus = []
    for idx, mem, is_idle in gpus:
        claim_dir = CLAIMS_DIR / f"gpu_{idx}.lock"
        if claim_dir.exists():
            # Already claimed, skip
            continue
        available_gpus.append((idx, mem, is_idle))

    # Sort: idle first, then by free memory descending
    available_gpus.sort(key=lambda x: (not x[2], -x[1]))

    # Try to claim num_gpus
    claimed_indices = []
    for idx, _, _ in available_gpus:
        if len(claimed_indices) >= num_gpus:
            break

        claim_dir = CLAIMS_DIR / f"gpu_{idx}.lock"
        try:
            # Atomic mkdir
            claim_dir.mkdir(mode=0o755)
            # Write metadata
            (claim_dir / "pid").write_text(str(pid))
            (claim_dir / "job_id").write_text(job_id)
            (claim_dir / "claimed_at").write_text(str(time.time()))
            claimed_indices.append(idx)
        except FileExistsError:
            # Lost race, another process claimed this GPU in between
            continue
        except Exception as e:
            print(f"Error claiming GPU {idx}: {e}", file=sys.stderr)
            # Rollback already-claimed GPUs
            for claimed_idx in claimed_indices:
                rollback_dir = CLAIMS_DIR / f"gpu_{claimed_idx}.lock"
                try:
                    shutil.rmtree(rollback_dir)
                except Exception:
                    pass
            return None

    if len(claimed_indices) == num_gpus:
        return ",".join(str(i) for i in sorted(claimed_indices))
    else:
        # Rollback
        for claimed_idx in claimed_indices:
            rollback_dir = CLAIMS_DIR / f"gpu_{claimed_idx}.lock"
            try:
                shutil.rmtree(rollback_dir)
            except Exception:
                pass
        return None


def release_gpus(job_id: str):
    """Remove all gpu_N.lock/ directories owned by job_id."""
    ensure_state_dirs()
    for claim_dir in CLAIMS_DIR.glob("gpu_*.lock"):
        try:
            job_id_file = claim_dir / "job_id"
            if job_id_file.exists() and job_id_file.read_text().strip() == job_id:
                shutil.rmtree(claim_dir)
        except Exception:
            pass


def clean_stale_claims():
    """Remove claims where PID is dead or older than 2 hours."""
    ensure_state_dirs()
    cutoff_time = time.time() - 2 * 3600  # 2 hours

    for claim_dir in CLAIMS_DIR.glob("gpu_*.lock"):
        try:
            pid_file = claim_dir / "pid"
            claimed_at_file = claim_dir / "claimed_at"

            if pid_file.exists():
                pid = int(pid_file.read_text().strip())
                if not pid_is_alive(pid):
                    shutil.rmtree(claim_dir)
                    continue

            if claimed_at_file.exists():
                claimed_at = float(claimed_at_file.read_text().strip())
                if claimed_at < cutoff_time:
                    shutil.rmtree(claim_dir)
        except Exception:
            pass


def update_job_status(
    job_id: str,
    status: str,
    pid: Optional[int] = None,
    gpus: Optional[str] = None,
    exit_code: Optional[int] = None,
    hostname: Optional[str] = None,
    script_path: Optional[str] = None,
    slurm_job_id: Optional[str] = None,
):
    """Update a job's status JSON file."""
    ensure_state_dirs()
    job_file = JOBS_DIR / f"{job_id}.json"

    if job_file.exists():
        job = json.loads(job_file.read_text())
    else:
        job = {
            "job_id": job_id,
            "status": "unknown",
            "queued_at": datetime.utcnow().isoformat() + "Z",
        }

    job["status"] = status
    if pid is not None:
        job["pid"] = pid
    if gpus is not None:
        job["gpus_claimed"] = gpus
    if exit_code is not None:
        job["exit_code"] = exit_code
    if hostname is not None:
        job["hostname"] = hostname
    if script_path is not None:
        job["script_path"] = script_path
    if slurm_job_id is not None:
        job["slurm_job_id"] = slurm_job_id

    if status == "running" and "started_at" not in job:
        job["started_at"] = datetime.utcnow().isoformat() + "Z"
    elif status in ("done", "failed") and "ended_at" not in job:
        job["ended_at"] = datetime.utcnow().isoformat() + "Z"

    job_file.write_text(json.dumps(job, indent=2))


def list_jobs() -> Dict:
    """Return all current jobs and claims as a dictionary."""
    ensure_state_dirs()
    jobs = {}
    claims = {}

    # Load all job files
    for job_file in JOBS_DIR.glob("*.json"):
        try:
            job = json.loads(job_file.read_text())
            jobs[job["job_id"]] = job
        except Exception:
            pass

    # Load all claims
    for claim_dir in CLAIMS_DIR.glob("gpu_*.lock"):
        try:
            gpu_idx = int(claim_dir.name[len("gpu_"):-len(".lock")])
            job_id_file = claim_dir / "job_id"
            if job_id_file.exists():
                job_id = job_id_file.read_text().strip()
                claims[f"gpu_{gpu_idx}"] = {
                    "job_id": job_id,
                    "pid": int((claim_dir / "pid").read_text().strip())
                    if (claim_dir / "pid").exists()
                    else None,
                }
        except Exception:
            pass

    return {"jobs": jobs, "claims": claims}


def cmd_claim(args):
    """Handle 'claim' subcommand."""
    result = claim_gpus(args.num_gpus, args.job_id, args.pid)
    if result is not None:
        print(result)
        # Update job status to waiting
        update_job_status(args.job_id, "waiting", pid=args.pid)
        return 0
    else:
        return 1


def cmd_release(args):
    """Handle 'release' subcommand."""
    release_gpus(args.job_id)
    return 0


def cmd_update_status(args):
    """Handle 'update-status' subcommand."""
    update_job_status(
        args.job_id,
        args.status,
        pid=args.pid,
        gpus=args.gpus,
        exit_code=args.exit_code,
        hostname=args.hostname,
        script_path=args.script_path,
        slurm_job_id=args.slurm_job_id,
    )
    return 0


def cmd_list(args):
    """Handle 'list' subcommand."""
    data = list_jobs()
    print(json.dumps(data, indent=2))
    return 0


def cmd_clean(args):
    """Handle 'clean' subcommand."""
    clean_stale_claims()

    # Optionally remove old job records
    ensure_state_dirs()
    cutoff = datetime.utcnow() - timedelta(hours=24)
    for job_file in JOBS_DIR.glob("*.json"):
        try:
            job = json.loads(job_file.read_text())
            if job["status"] in ("done", "failed"):
                ended_at = job.get("ended_at")
                if ended_at:
                    ended = datetime.fromisoformat(ended_at.replace("Z", "+00:00")).replace(tzinfo=None)
                    if ended < cutoff:
                        job_file.unlink()
        except Exception:
            pass
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Atomic GPU claim/release management."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # claim
    claim_parser = subparsers.add_parser("claim", help="Claim N GPUs")
    claim_parser.add_argument(
        "--num-gpus", type=int, required=True, help="Number of GPUs to claim"
    )
    claim_parser.add_argument(
        "--job-id", type=str, required=True, help="Job identifier"
    )
    claim_parser.add_argument("--pid", type=int, required=True, help="Process PID")
    claim_parser.set_defaults(func=cmd_claim)

    # release
    release_parser = subparsers.add_parser("release", help="Release claimed GPUs")
    release_parser.add_argument("--job-id", type=str, required=True, help="Job ID")
    release_parser.set_defaults(func=cmd_release)

    # update-status
    status_parser = subparsers.add_parser("update-status", help="Update job status")
    status_parser.add_argument("--job-id", type=str, required=True, help="Job ID")
    status_parser.add_argument("--status", type=str, required=True, help="New status")
    status_parser.add_argument("--pid", type=int, default=None, help="Process PID")
    status_parser.add_argument("--gpus", type=str, default=None, help="GPU indices")
    status_parser.add_argument(
        "--exit-code", type=int, default=None, help="Exit code"
    )
    status_parser.add_argument(
        "--hostname", type=str, default=None, help="Host running the job"
    )
    status_parser.add_argument(
        "--script-path", type=str, default=None, help="Script launched for the job"
    )
    status_parser.add_argument(
        "--slurm-job-id", type=str, default=None, help="Parent Slurm job allocation id"
    )
    status_parser.set_defaults(func=cmd_update_status)

    # list
    list_parser = subparsers.add_parser("list", help="List all jobs and claims")
    list_parser.set_defaults(func=cmd_list)

    # clean
    clean_parser = subparsers.add_parser(
        "clean", help="Remove stale claims and old job records"
    )
    clean_parser.set_defaults(func=cmd_clean)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
