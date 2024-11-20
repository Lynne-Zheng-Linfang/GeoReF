#!/usr/bin/env bash
this_dir=$(dirname "$0")
# commonly used opts:

# MODEL.WEIGHTS: resume or pretrained, or test checkpoint
CFG=$1
CUDA_VISIBLE_DEVICES=$2
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
NGPU=${#GPUS[@]}
echo "use gpu ids: $CUDA_VISIBLE_DEVICES num gpus: $NGPU"
CKPT=$3

# Check if the checkpoint file exists
if [ ! -f "$CKPT" ]; then
    echo "$CKPT does not exist."
    exit 1
fi

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

set -x
NCCL_DEBUG=INFO
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
PYTHONPATH="$this_dir/../..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python $this_dir/main_georef.py \
    --config-file $CFG --num-gpus $NGPU --eval-only \
    --seed $SEED \
    --opts MODEL.WEIGHTS=$CKPT \
    ${@:4}