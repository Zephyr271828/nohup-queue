"""Path resolution and job record I/O.

State directory priority:
  $NOHUP_QUEUE_CACHE_DIR/.gpu_queue   (if env var is set)
  $HOME/.gpu_queue                    (fallback)
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


def get_state_dir() -> Path:
    """Return the queue state directory."""
    cache = os.environ.get("NOHUP_QUEUE_CACHE_DIR")
    if cache:
        return Path(cache) / ".gpu_queue"
    return Path.home() / ".gpu_queue"


def get_jobs_dir() -> Path:
    return get_state_dir() / "jobs"


def get_claims_dir() -> Path:
    return get_state_dir() / "claims"


def ensure_state_dirs() -> None:
    get_jobs_dir().mkdir(parents=True, exist_ok=True)
    get_claims_dir().mkdir(parents=True, exist_ok=True)


def load_jobs() -> Dict[str, Dict]:
    """Load all job records from JSON files."""
    jobs: Dict[str, Dict] = {}
    jobs_dir = get_jobs_dir()
    if not jobs_dir.exists():
        return jobs
    for job_file in jobs_dir.glob("*.json"):
        try:
            job = json.loads(job_file.read_text())
            jobs[job.get("job_id", "unknown")] = job
        except Exception:
            pass
    return jobs


def update_job_status(
    job_id: str,
    status: str,
    *,
    pid: Optional[int] = None,
    gpus: Optional[str] = None,
    exit_code: Optional[int] = None,
    hostname: Optional[str] = None,
    script_path: Optional[str] = None,
    slurm_job_id: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """Write or update a job's JSON record."""
    ensure_state_dirs()
    job_file = get_jobs_dir() / f"{job_id}.json"

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
    if log_file is not None:
        job["log_file"] = log_file

    if status == "running" and "started_at" not in job:
        job["started_at"] = datetime.utcnow().isoformat() + "Z"
    elif status in ("done", "failed", "interrupted") and "ended_at" not in job:
        job["ended_at"] = datetime.utcnow().isoformat() + "Z"

    job_file.write_text(json.dumps(job, indent=2))
