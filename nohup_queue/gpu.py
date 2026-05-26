"""Atomic GPU claim/release management.

Uses os.mkdir() as atomic primitive (safe on WekaFS/NFS).
Each GPU is represented by a directory gpu_<host>_<idx>.lock/ in the claims dir.

Claim policy: a GPU is claimable when it has enough free VRAM and no foreign
compute process is holding more than a small "decoration" budget. Utilization
percentage is intentionally ignored — keep-alive scripts that drive SMs to
~100% on a few MB of VRAM (e.g. to keep a SLURM job from being reaped) must
not block real training claims.

Tunable via environment:
  NOHUP_QUEUE_IDLE_FREE_RATIO  fraction of total VRAM that must be free
                               (default 0.5)
  NOHUP_QUEUE_IDLE_FREE_MIB    absolute free-VRAM floor in MiB, overrides
                               ratio when set (default unset)
  NOHUP_QUEUE_IDLE_PROC_MIB    largest foreign single-process VRAM that
                               is still tolerated (default 4096)
  NOHUP_QUEUE_STALE_LOCK_SECS  age in seconds after which a foreign-host
                               or malformed lock is reaped (default 600)
"""

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .state import get_claims_dir, ensure_state_dirs


_HOSTNAME = socket.gethostname()


def _idle_free_ratio() -> float:
    try:
        return float(os.environ.get("NOHUP_QUEUE_IDLE_FREE_RATIO", "0.5"))
    except ValueError:
        return 0.5


def _idle_free_mib_abs() -> Optional[float]:
    raw = os.environ.get("NOHUP_QUEUE_IDLE_FREE_MIB")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _idle_proc_mib() -> float:
    try:
        return float(os.environ.get("NOHUP_QUEUE_IDLE_PROC_MIB", "4096"))
    except ValueError:
        return 4096.0


def _stale_lock_secs() -> float:
    try:
        return float(os.environ.get("NOHUP_QUEUE_STALE_LOCK_SECS", "600"))
    except ValueError:
        return 600.0


def _lock_name(idx: int) -> str:
    """Lock dir name for GPU `idx` on the current host."""
    return f"gpu_{_HOSTNAME}_{idx}.lock"


def _parse_lock_name(name: str) -> Optional[Tuple[str, int]]:
    """Return (host, idx) for a lock dir name, or None if it doesn't parse.

    Accepts the new host-scoped form ``gpu_<host>_<idx>.lock``. Legacy
    unscoped names (``gpu_<idx>.lock``) are reported as host="" so they
    can be reaped by clean_stale_claims.
    """
    if not (name.startswith("gpu_") and name.endswith(".lock")):
        return None
    body = name[len("gpu_"):-len(".lock")]
    # Try host-scoped form first.
    rsplit = body.rsplit("_", 1)
    if len(rsplit) == 2:
        host, idx_s = rsplit
        try:
            return host, int(idx_s)
        except ValueError:
            pass
    # Legacy unscoped form.
    try:
        return "", int(body)
    except ValueError:
        return None


def pid_is_alive(pid: int) -> bool:
    """Return True if the process exists."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False


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


def _max_proc_mib_by_gpu() -> Dict[int, float]:
    """Return per-GPU max single-process VRAM (MiB).

    Used as the "is this GPU effectively free?" signal. A GPU running only
    tiny keep-alive processes (< NOHUP_QUEUE_IDLE_PROC_MIB) is still claimable.
    """
    worst: Dict[int, float] = {}
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-compute-apps=gpu_uuid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return worst
        uuid_map = gpu_uuid_to_index()
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            gpu_uuid, mib_s = parts
            idx = uuid_map.get(gpu_uuid)
            if idx is None:
                continue
            try:
                mib = float(mib_s)
            except ValueError:
                continue
            if mib > worst.get(idx, 0.0):
                worst[idx] = mib
    except Exception:
        pass
    return worst


def get_gpu_list() -> List[Tuple[int, float, bool]]:
    """Return list of (gpu_idx, free_memory_MiB, is_claimable).

    `is_claimable` is True when:
      * free VRAM >= floor (absolute MiB env, else ratio*total), AND
      * no single compute process holds more than NOHUP_QUEUE_IDLE_PROC_MIB.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []

        gpu_mem: Dict[int, Tuple[float, float]] = {}
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                try:
                    gpu_mem[int(parts[0])] = (float(parts[1]), float(parts[2]))
                except ValueError:
                    pass

        proc_worst = _max_proc_mib_by_gpu()
        free_abs = _idle_free_mib_abs()
        free_ratio = _idle_free_ratio()
        proc_cap = _idle_proc_mib()

        gpus: List[Tuple[int, float, bool]] = []
        for idx in sorted(gpu_mem.keys()):
            free, total = gpu_mem[idx]
            floor = free_abs if free_abs is not None else free_ratio * total
            claimable = (free >= floor) and (proc_worst.get(idx, 0.0) <= proc_cap)
            gpus.append((idx, free, claimable))
        return gpus
    except Exception:
        return []


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


def _claim_age(claim_dir: Path) -> Optional[float]:
    """Seconds since the claim was created, or None if unknown."""
    try:
        ts_file = claim_dir / "claimed_at"
        if ts_file.exists():
            return time.time() - float(ts_file.read_text().strip())
    except Exception:
        pass
    try:
        return time.time() - claim_dir.stat().st_mtime
    except Exception:
        return None


def clean_stale_claims() -> None:
    """Remove claims that are no longer authoritative.

    A lock is stale if any of:
      * it is malformed (no/invalid pid file, unparseable name),
      * its host matches this host AND the owner PID is dead,
      * its host does NOT match this host AND it is older than
        NOHUP_QUEUE_STALE_LOCK_SECS (defends against locks left behind
        on a shared filesystem when an allocation on another node ends).
    """
    ensure_state_dirs()
    stale_secs = _stale_lock_secs()
    for claim_dir in get_claims_dir().glob("gpu_*.lock"):
        try:
            parsed = _parse_lock_name(claim_dir.name)
            pid_file = claim_dir / "pid"
            pid: Optional[int] = None
            if pid_file.exists():
                try:
                    pid = int(pid_file.read_text().strip())
                except (ValueError, OSError):
                    pid = None

            # 1. Malformed name or missing/unparseable pid file → stale.
            if parsed is None or pid is None:
                age = _claim_age(claim_dir) or 0.0
                # Be a little patient with brand-new claims: there's a
                # narrow window after mkdir() and before pid is written
                # where a concurrent cleaner would otherwise nuke a
                # legitimately-in-progress claim. Anything older than
                # ~5s with no pid file is definitively broken.
                if age > 5.0:
                    shutil.rmtree(claim_dir)
                continue

            host, _idx = parsed

            # 2. Foreign-host lock: PID liveness here is meaningless
            #    (different namespace). Reap by age only.
            if host and host != _HOSTNAME:
                age = _claim_age(claim_dir)
                if age is not None and age > stale_secs:
                    shutil.rmtree(claim_dir)
                continue

            # 3. Our host (or legacy unscoped): the PID must be alive.
            if not pid_is_alive(pid):
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
        (idx, mem, claimable) for idx, mem, claimable in gpus
        if claimable and not (claims_dir / _lock_name(idx)).exists()
    ]
    # Most free memory first
    available.sort(key=lambda x: -x[1])

    claimed_indices: List[int] = []
    for idx, _, _ in available:
        if len(claimed_indices) >= num_gpus:
            break
        claim_dir = claims_dir / _lock_name(idx)
        try:
            claim_dir.mkdir(mode=0o755)
            # Write pid first so a crash in the middle still leaves a
            # cleanable lock (clean_stale_claims keys on pid liveness).
            (claim_dir / "pid").write_text(str(pid))
            (claim_dir / "job_id").write_text(job_id)
            (claim_dir / "host").write_text(_HOSTNAME)
            (claim_dir / "claimed_at").write_text(str(time.time()))
            claimed_indices.append(idx)
        except FileExistsError:
            continue
        except Exception:
            for ci in claimed_indices:
                try:
                    shutil.rmtree(claims_dir / _lock_name(ci))
                except Exception:
                    pass
            return None

    if len(claimed_indices) == num_gpus:
        return ",".join(str(i) for i in sorted(claimed_indices))

    # Not enough GPUs — roll back
    for ci in claimed_indices:
        try:
            shutil.rmtree(claims_dir / _lock_name(ci))
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
    """Return active GPU claims on the current host: gpu_idx -> (job_id, pid).

    Foreign-host locks are filtered out because their PIDs can't be checked
    locally and their GPU indices live in a different namespace.
    """
    claims: Dict[int, Tuple[str, int]] = {}
    claims_dir = get_claims_dir()
    if not claims_dir.exists():
        return claims
    for claim_dir in claims_dir.glob("gpu_*.lock"):
        try:
            parsed = _parse_lock_name(claim_dir.name)
            if parsed is None:
                continue
            host, gpu_idx = parsed
            if host and host != _HOSTNAME:
                continue
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
