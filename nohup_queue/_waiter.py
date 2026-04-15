"""Background waiter process.

Invoked as:
  python -m nohup_queue._waiter \\
    job_id script_path pids num_gpus hostname slurm_job_id cwd

stdout/stderr are redirected to the log file by the parent process (nq).
"""

import os
import signal
import subprocess
import sys
import time
from datetime import datetime


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main() -> int:
    args = sys.argv[1:]
    if len(args) != 7:
        print(
            "Usage: python -m nohup_queue._waiter "
            "job_id script_path pids num_gpus hostname slurm_job_id cwd",
            file=sys.stderr,
        )
        return 1

    job_id, script_path, pids_str, num_gpus_str, hostname, slurm_job_id, cwd = args
    num_gpus = int(num_gpus_str)

    # Run from the directory where nq was invoked so relative paths in the
    # script still work.
    if cwd:
        os.chdir(cwd)

    # Deferred imports so the package does not need to be importable before
    # the process is fully detached.
    from .gpu import claim_gpus, clean_stale_claims, pid_is_alive, release_gpus
    from .state import update_job_status

    finalized = [False]

    def finalize(status: str, exit_code: int) -> None:
        if finalized[0]:
            return
        finalized[0] = True
        try:
            update_job_status(
                job_id, status,
                exit_code=exit_code,
                hostname=hostname,
                script_path=script_path,
                slurm_job_id=slurm_job_id or None,
            )
        except Exception:
            pass

    def handle_signal(signum, frame) -> None:
        finalize("interrupted", 128)
        try:
            release_gpus(job_id)
        except Exception:
            pass
        sys.exit(128)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGHUP, handle_signal)

    # --- Wait for prerequisite PIDs ---
    if pids_str:
        for pid_s in pids_str.split(","):
            pid_s = pid_s.strip()
            if not pid_s:
                continue
            try:
                dep_pid = int(pid_s)
            except ValueError:
                continue
            while pid_is_alive(dep_pid):
                log(f"Waiting for PID {dep_pid}...")
                time.sleep(5)
            log(f"PID {dep_pid} finished.")

    # --- GPU polling ---
    if num_gpus > 0:
        while True:
            claimed = claim_gpus(num_gpus, job_id, os.getpid())
            if claimed is not None:
                log(f"Claimed GPUs: {claimed}")
                os.environ["CUDA_VISIBLE_DEVICES"] = claimed
                update_job_status(
                    job_id, "running",
                    pid=os.getpid(),
                    gpus=claimed,
                    hostname=hostname,
                    script_path=script_path,
                    slurm_job_id=slurm_job_id or None,
                )
                break
            clean_stale_claims()
            log(f"Waiting for {num_gpus} GPUs (retrying in 30s)...")
            time.sleep(30)
    else:
        log("No GPU waiting required (NUM_GPUS=0)")
        update_job_status(
            job_id, "running",
            pid=os.getpid(),
            hostname=hostname,
            script_path=script_path,
            slurm_job_id=slurm_job_id or None,
        )

    # --- Run the script ---
    log(f"Starting: bash {script_path}")
    env = os.environ.copy()
    env["NOHUP_JOB_ID"] = job_id

    exit_code = 0
    try:
        result = subprocess.run(["bash", script_path], env=env)
        exit_code = result.returncode
    except Exception as exc:
        log(f"Error running script: {exc}")
        exit_code = 1

    if exit_code == 0:
        finalize("done", 0)
    else:
        finalize("failed", exit_code)

    try:
        release_gpus(job_id)
    except Exception:
        pass

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
