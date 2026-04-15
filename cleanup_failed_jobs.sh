#!/bin/bash
set -euo pipefail

queue_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/nohup-queue" && pwd)"

echo "Releasing GPU claims for all failed jobs..."
if [[ -d .gpu_queue/jobs ]]; then
  for job_file in .gpu_queue/jobs/job_*.json; do
    [[ -f "$job_file" ]] || continue
    job_id=$(basename "$job_file" .json)
    # Use python to safely parse JSON
    status=$(python3 -c "import json; j=json.load(open('$job_file')); print(j.get('status', ''))" 2>/dev/null || echo "")
    if [[ "$status" == "failed" ]]; then
      echo "  Releasing $job_id..."
      python3 "$queue_dir/gpu_claim.py" release --job-id "$job_id" 2>/dev/null || true
    fi
  done
fi

echo "Cleaning stale claims..."
python3 "$queue_dir/gpu_claim.py" clean

echo "Removing failed job records..."
if [[ -d .gpu_queue/jobs ]]; then
  for job_file in .gpu_queue/jobs/job_*.json; do
    [[ -f "$job_file" ]] || continue
    status=$(python3 -c "import json; j=json.load(open('$job_file')); print(j.get('status', ''))" 2>/dev/null || echo "")
    if [[ "$status" == "failed" ]]; then
      rm -f "$job_file"
      echo "  Removed $job_file"
    fi
  done
fi

echo ""
echo "Done. Remaining jobs:"
./njobs
