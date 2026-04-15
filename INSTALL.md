# Install `nohup-queue`

Keep this simple.

You only need these files:

- `nohup_run.sh`
- `gpu_claim.py`
- `njobs`

## Option 1: Copy into your project

```bash
cp nohup_run.sh /path/to/project/
cp gpu_claim.py /path/to/project/
cp njobs /path/to/project/

chmod +x /path/to/project/nohup_run.sh
chmod +x /path/to/project/njobs
```

Then run from your project root:

```bash
NUM_GPUS=8 ./nohup_run.sh scripts/train.sh
./njobs
```

## Option 2: Keep this folder and call it directly

```bash
NUM_GPUS=8 /path/to/nohup-queue/nohup_run.sh scripts/train.sh
/path/to/nohup-queue/njobs
```

You can also make symlinks if you want shorter commands:

```bash
ln -s /path/to/nohup-queue/nohup_run.sh nohup_run.sh
ln -s /path/to/nohup-queue/njobs njobs
```

## Ignore queue state

The scripts create `.gpu_queue/` in the directory where you run them.

Add this to your project's `.gitignore`:

```bash
echo ".gpu_queue/" >> .gitignore
```

You may also want:

```bash
echo "logs/" >> .gitignore
```

## Requirements

- `bash`
- `python3`
- `nvidia-smi`

No extra Python packages are needed.

## Quick check

```bash
python3 gpu_claim.py --help
./njobs --help
NUM_GPUS=0 ./nohup_run.sh scripts/eval.sh
```

If you are using the direct path version, replace `./` with the full path.

## Important

If your training script sets `CUDA_VISIBLE_DEVICES` by itself, it should not overwrite a value that is already set by `nohup_run.sh`.

Use this pattern:

```bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(get_free_gpus ${NUM_GPUS})}
```
