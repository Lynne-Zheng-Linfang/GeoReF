#!/usr/bin/env bash
this_dir=$(dirname "$0")
# commonly used opts:

# MODEL.WEIGHTS: resume or pretrained, or test checkpoint
CFG=$1
CUDA_VISIBLE_DEVICES=$2
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
NGPU=${#GPUS[@]}
echo "use gpu ids: $CUDA_VISIBLE_DEVICES num gpus: $NGPU"

# Initialize seed
SEED=-1
# Extract seed value if provided
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed)
            SEED=$2
            echo "Using seed: $SEED"
            shift
            ;;
        *)
            ;;
    esac
    shift
done

# Reset positional parameters to handle arguments correctly
set -- "$CFG" "$CUDA_VISIBLE_DEVICES" "${@:3}"

set -x
# CUDA_LAUNCH_BLOCKING=1
NCCL_DEBUG=INFO
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
PYTHONPATH="$this_dir/../..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$2 python $this_dir/main_georef.py \
    --config-file $CFG --num-gpus $NGPU ${SEED:+--seed=$SEED} "${@:3}"