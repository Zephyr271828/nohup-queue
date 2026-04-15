"""Job launching — the core of the 'nq' command."""

import os
import secrets
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from .state import ensure_state_dirs, update_job_status


def launch_job(
    script_path: str,
    pids: str = "",
    num_gpus: int = 0,
    log_dir: Optional[str] = None,
) -> None:
    """Queue a script as a detached background job.

    Args:
        script_path: Path to the bash script to run.
        pids:        Comma-separated PIDs to wait for before starting.
        num_gpus:    Number of GPUs to claim (0 = no GPU gating).
        log_dir:     Override log directory. Default: ./logs/<task_name>/
    """
    hostname = socket.gethostname()
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    task_name = Path(script_path).stem
    cwd = str(Path.cwd())

    # Build log file path (kept per-project, not in the global state dir)
    if log_dir is None:
        log_parent = Path.cwd() / "logs" / task_name
    else:
        log_parent = Path(log_dir)
    log_parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_parent / f"{timestamp}.log"
    while log_file.exists():
        time.sleep(1)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_parent / f"{timestamp}.log"

    job_id = f"job_{secrets.token_hex(4)}"
    script_abs = str(Path(script_path).resolve())

    # Register the job as pending before launching the waiter, so njobs
    # shows it immediately even if the waiter hasn't started yet.
    ensure_state_dirs()
    update_job_status(
        job_id, "pending",
        hostname=hostname,
        script_path=script_abs,
        slurm_job_id=slurm_job_id or None,
        log_file=str(log_file),
    )

    waiter_cmd = [
        sys.executable, "-m", "nohup_queue._waiter",
        job_id,
        script_abs,
        pids or "",
        str(num_gpus),
        hostname,
        slurm_job_id or "",
        cwd,
    ]

    with open(log_file, "a") as lf:
        proc = subprocess.Popen(
            waiter_cmd,
            stdin=subprocess.DEVNULL,
            stdout=lf,
            stderr=lf,
            start_new_session=True,  # detach from terminal
            close_fds=True,
        )

    print(f"Job ID:     {job_id}")
    print(f"Log:        {log_file}")
    print(f"Waiter PID: {proc.pid}")
    print("Status:     pending (use 'njobs' to check)")
