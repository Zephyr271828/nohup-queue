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
- tracks job status with `nohup-queue jobs`

## Commands

A single `nohup-queue` entry point with three subcommands:

- `nohup-queue run <script>`: submit a job
- `nohup-queue jobs`: show queue and GPU status
- `nohup-queue clean`: remove stale claims and dead job records

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
nohup-queue run scripts/train_a.sh --num-gpus 8
nohup-queue run scripts/train_b.sh --num-gpus 8
nohup-queue run scripts/eval.sh --num-gpus 4
```

`NUM_GPUS` is also honored as the default for `--num-gpus`:

```bash
NUM_GPUS=8 nohup-queue run scripts/train.sh
```

Each job runs in the background with `nohup`. If the GPUs are not free yet, it waits and starts later.

Check status:

```bash
nohup-queue jobs
```

## Typical workflow

1. Get onto a GPU node.
2. Start `tmux` or `screen`.
3. Queue several jobs with `nohup-queue run`.
4. Disconnect if you want.
5. Reconnect later and run `nohup-queue jobs`.

This is useful when the node is valuable and you want your next job to start immediately after the current one finishes.

## Command examples

Run a training job that needs 8 GPUs:

```bash
nohup-queue run scripts/train.sh --num-gpus 8
```

Run without GPU waiting:

```bash
nohup-queue run scripts/eval.sh --num-gpus 0
```

Wait for another local process first:

```bash
nohup-queue run scripts/train.sh --num-gpus 8 --pids 12345
```

Show all remembered jobs (including done):

```bash
nohup-queue jobs --all
```

Clean stale state:

```bash
nohup-queue clean
```

Also drop done/failed records older than 24h:

```bash
nohup-queue clean --clear-done
```

## Notes

- Queue state is stored in `$HOME/.gpu_queue/` by default, or `$NOHUP_QUEUE_CACHE_DIR/.gpu_queue/` if set.
- Logs default to `./logs/<task_name>/`; override with `--log-dir`.
- This is meant for one node at a time.
- Your script should respect `CUDA_VISIBLE_DEVICES` if it is already set.
