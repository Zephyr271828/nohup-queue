"""Atomic GPU claim/release management.

Uses os.mkdir() as atomic primitive (safe on WekaFS/NFS).
Each GPU is represented by a directory gpu_N.lock/ in the claims dir.
"""

import os
import shutil
import subprocess
import time
from typing import Dict, List, Optional, Tuple

from .state import get_claims_dir, ensure_state_dirs


def pid_is_alive(pid: int) -> bool:
    """Return True if the process exists."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False


def get_gpu_list() -> List[Tuple[int, float, bool]]:
    """Return list of (gpu_idx, free_memory_MiB, is_idle)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []

        gpu_mem: Dict[int, float] = {}
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split(",")
            if len(parts) == 2:
                try:
                    gpu_mem[int(parts[0].strip())] = float(parts[1].strip())
                except ValueError:
                    pass

        gpus = []
        for idx in sorted(gpu_mem.keys()):
            is_idle = True
            try:
                r = subprocess.run(
                    ["nvidia-smi", "-i", str(idx),
                     "--query-compute-apps=pid",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                )
                if r.returncode == 0:
                    out = r.stdout.strip()
                    if out and out != "No running processes found":
                        is_idle = False
            except Exception:
                pass
            gpus.append((idx, gpu_mem[idx], is_idle))

        return gpus
    except Exception:
        return []


def gpu_uuid_to_index() -> Dict[str, int]:
    """Return mapping from GPU UUID to index."""
    mapping: Dict[str, int] = {}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return mapping
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = [p.strip() for p in line.split(",", 1)]
            if len(parts) == 2:
                try:
                    mapping[parts[1]] = int(parts[0])
                except ValueError:
                    pass
    except Exception:
        pass
    return mapping


def live_gpu_processes() -> Dict[int, List[int]]:
    """Return dict of gpu_idx -> list of live compute PIDs."""
    gpu_pids: Dict[int, List[int]] = {}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return gpu_pids
        uuid_map = gpu_uuid_to_index()
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            gpu_uuid, pid_str = parts
            try:
                pid = int(pid_str)
            except ValueError:
                continue
            gpu_idx = uuid_map.get(gpu_uuid)
            if gpu_idx is None or not pid_is_alive(pid):
                continue
            gpu_pids.setdefault(gpu_idx, []).append(pid)
    except Exception:
        pass
    return gpu_pids


def clean_stale_claims() -> None:
    """Remove claims where the owner PID is dead or claim is >2h old."""
    ensure_state_dirs()
    cutoff = time.time() - 2 * 3600
    for claim_dir in get_claims_dir().glob("gpu_*.lock"):
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
                if claimed_at < cutoff:
                    shutil.rmtree(claim_dir)
        except Exception:
            pass


def claim_gpus(num_gpus: int, job_id: str, pid: int) -> Optional[str]:
    """Atomically claim num_gpus GPUs.

    Returns comma-separated GPU indices on success, None on failure.
    On failure, rolls back any partial claims.
    """
    if num_gpus == 0:
        return ""

    ensure_state_dirs()
    clean_stale_claims()

    gpus = get_gpu_list()
    claims_dir = get_claims_dir()

    available = [
        (idx, mem, idle) for idx, mem, idle in gpus
        if not (claims_dir / f"gpu_{idx}.lock").exists()
    ]
    # Prefer idle GPUs, then most free memory
    available.sort(key=lambda x: (not x[2], -x[1]))

    claimed_indices: List[int] = []
    for idx, _, _ in available:
        if len(claimed_indices) >= num_gpus:
            break
        claim_dir = claims_dir / f"gpu_{idx}.lock"
        try:
            claim_dir.mkdir(mode=0o755)
            (claim_dir / "pid").write_text(str(pid))
            (claim_dir / "job_id").write_text(job_id)
            (claim_dir / "claimed_at").write_text(str(time.time()))
            claimed_indices.append(idx)
        except FileExistsError:
            continue
        except Exception:
            for ci in claimed_indices:
                try:
                    shutil.rmtree(claims_dir / f"gpu_{ci}.lock")
                except Exception:
                    pass
            return None

    if len(claimed_indices) == num_gpus:
        return ",".join(str(i) for i in sorted(claimed_indices))

    # Not enough GPUs — roll back
    for ci in claimed_indices:
        try:
            shutil.rmtree(claims_dir / f"gpu_{ci}.lock")
        except Exception:
            pass
    return None


def release_gpus(job_id: str) -> None:
    """Remove all GPU lock directories owned by this job."""
    ensure_state_dirs()
    for claim_dir in get_claims_dir().glob("gpu_*.lock"):
        try:
            job_id_file = claim_dir / "job_id"
            if job_id_file.exists() and job_id_file.read_text().strip() == job_id:
                shutil.rmtree(claim_dir)
        except Exception:
            pass


def load_claims() -> Dict[int, Tuple[str, int]]:
    """Return active GPU claims: gpu_idx -> (job_id, pid)."""
    claims: Dict[int, Tuple[str, int]] = {}
    claims_dir = get_claims_dir()
    if not claims_dir.exists():
        return claims
    for claim_dir in claims_dir.glob("gpu_*.lock"):
        try:
            gpu_idx = int(claim_dir.name[len("gpu_"):-len(".lock")])
            job_id_file = claim_dir / "job_id"
            pid_file = claim_dir / "pid"
            if job_id_file.exists() and pid_file.exists():
                job_id = job_id_file.read_text().strip()
                pid = int(pid_file.read_text().strip())
                if pid_is_alive(pid):
                    claims[gpu_idx] = (job_id, pid)
        except Exception:
            pass
    return claims
