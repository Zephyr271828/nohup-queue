#!/bin/bash

set -euo pipefail

script_path=$1
shift

# Optional: --pids <pid1,pid2,...>
pids=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pids)
            pids="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

task_name=$(basename ${script_path} .sh)
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="$(pwd)/logs/${task_name}/${timestamp}.log"
job_id="job_$(openssl rand -hex 4)"

mkdir -p "$(pwd)/logs/${task_name}"

# Get NUM_GPUS from environment (default 0 = no GPU waiting)
NUM_GPUS=${NUM_GPUS:-0}

# Determine queue directory (where nohup_run.sh and gpu_claim.py are).
# Resolve symlinks so invoking ./nohup_run.sh from repo root still finds helpers.
queue_dir="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

# Build the waiter script that will run in background
waiter_script=$(mktemp /tmp/waiter_XXXXXX.sh)
cat > "$waiter_script" <<'WAITER_SCRIPT'
#!/bin/bash
set -euo pipefail

# Source the actual values from outer scope
job_id="$1"
script_path="$2"
log_file="$3"
pids="$4"
num_gpus="$5"
script_dir="$6"

# Function to run cleanup on exit
cleanup() {
    python3 "${script_dir}/gpu_claim.py" release --job-id "$job_id" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for prerequisite PIDs if given
if [[ -n "$pids" ]]; then
    IFS=',' read -ra pid_list <<< "$pids"
    for pid in "${pid_list[@]}"; do
        while kill -0 "$pid" 2>/dev/null; do
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for PID $pid..." >> "$log_file"
            sleep 5
        done
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] PID $pid finished." >> "$log_file"
    done
fi

# GPU polling loop
if [[ $num_gpus -gt 0 ]]; then
    while true; do
        claimed=$(python3 "${script_dir}/gpu_claim.py" claim \
                  --num-gpus "$num_gpus" --job-id "$job_id" --pid $$ 2>/dev/null)
        if [[ $? -eq 0 ]]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Claimed GPUs: $claimed" >> "$log_file"
            export CUDA_VISIBLE_DEVICES="$claimed"
            python3 "${script_dir}/gpu_claim.py" update-status \
                --job-id "$job_id" --status running --pid $$ --gpus "$claimed" 2>/dev/null || true
            break
        fi
        python3 "${script_dir}/gpu_claim.py" clean 2>/dev/null || true
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for $num_gpus GPUs (retrying in 30s)..." >> "$log_file"
        sleep 30
    done
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] No GPU waiting required (NUM_GPUS=0)" >> "$log_file"
    python3 "${script_dir}/gpu_claim.py" update-status \
        --job-id "$job_id" --status running --pid $$ 2>/dev/null || true
fi

# Run the actual training/eval script
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: bash $script_path" >> "$log_file"
bash "$script_path"
exit_code=$?

# Update final status
if [[ $exit_code -eq 0 ]]; then
    python3 "${script_dir}/gpu_claim.py" update-status \
        --job-id "$job_id" --status done --exit-code $exit_code 2>/dev/null || true
else
    python3 "${script_dir}/gpu_claim.py" update-status \
        --job-id "$job_id" --status failed --exit-code $exit_code 2>/dev/null || true
fi

exit $exit_code
WAITER_SCRIPT

chmod +x "$waiter_script"

# Launch the waiter script in background
nohup bash "$waiter_script" "$job_id" "$script_path" "$log_file" "$pids" "$NUM_GPUS" "$queue_dir" 2>&1 | tee -a "$log_file" > /dev/null 2>&1 &
waiter_pid=$!

echo "Job ID: $job_id"
echo "Log: $log_file"
echo "Waiter PID: $waiter_pid"
echo "Status: pending (use 'njobs' to check)"
