# nohup-queue

Small helper for running jobs on a GPU node you already have.

## Why this exists

On some systems, getting onto a GPU node is the hard part.

Once you finally get one, you usually want to keep that node busy instead of:

- sitting there waiting for one job to finish before starting the next one
- losing track of which GPUs are free
- accidentally starting two jobs on the same GPUs

`nohup-queue` is a simple local queue for that case. It is not a cluster scheduler. It is just a way to stay on one node, keep that node occupied, and launch jobs safely in the background.

Use it from a long-lived shell on that node:

- `tmux`
- `screen`
- `nohup`

## What it does

- waits until the requested number of GPUs are free
- claims those GPUs so other queued jobs do not race with it
- starts your script with `CUDA_VISIBLE_DEVICES` set
- tracks job status with `njobs`

## Files

- `nohup_run.sh`: submit a job
- `gpu_claim.py`: claim/release GPUs
- `njobs`: show queue status

## Basic use

Start a persistent session on the GPU node:

```bash
tmux new -s gpu-queue
```

or:

```bash
screen -S gpu-queue
```

Then submit jobs:

```bash
NUM_GPUS=8 ./nohup_run.sh scripts/train_a.sh
NUM_GPUS=8 ./nohup_run.sh scripts/train_b.sh
NUM_GPUS=4 ./nohup_run.sh scripts/eval.sh
```

Each job runs in the background with `nohup`. If the GPUs are not free yet, it waits and starts later.

Check status:

```bash
./njobs
```

## Typical workflow

1. Get onto a GPU node.
2. Start `tmux` or `screen`.
3. Queue several jobs with `./nohup_run.sh`.
4. Disconnect if you want.
5. Reconnect later and run `./njobs`.

This is useful when the node is valuable and you want your next job to start immediately after the current one finishes.

## Command examples

Run a training job that needs 8 GPUs:

```bash
NUM_GPUS=8 ./nohup_run.sh scripts/train.sh
```

Run without GPU waiting:

```bash
NUM_GPUS=0 ./nohup_run.sh scripts/eval.sh
```

Wait for another local process first:

```bash
NUM_GPUS=8 ./nohup_run.sh scripts/train.sh --pids 12345
```

Show all remembered jobs:

```bash
./njobs --all
```

Clean stale state:

```bash
./njobs --clean
```

## Notes

- Queue state is stored in `.gpu_queue/` in the current working directory.
- This is meant for one node at a time.
- Your script should respect `CUDA_VISIBLE_DEVICES` if it is already set.
