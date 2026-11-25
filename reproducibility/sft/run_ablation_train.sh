#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# GPU devices to use
GPU_DEVICES=(6)

# Seeds to iterate over
SEEDS=(888 999)

# Base configuration
MODEL_NAME="nvidia/Mistral-NeMo-Minitron-8B-Base"
NUM_EPOCHS=4
MAX_LENGTH=6144
MAX_NEW_TOKENS=2048
BASE_OUTPUT_DIR="${GIFARC_TRAIN_OUTPUT_DIR:-./models}"

# Ablation types
ABLATION_TYPES=("base" "analogy" "code" "full")

echo "Starting ablation study training with multiple seeds on GPU ${GPU_DEVICES[@]}"
echo "=================================================="

# Create array of all training jobs
JOBS=()
for SEED in "${SEEDS[@]}"; do
    for ABLATION_TYPE in "${ABLATION_TYPES[@]}"; do
        JOBS+=("${SEED}:${ABLATION_TYPE}")
    done
done

# Total number of jobs
TOTAL_JOBS=${#JOBS[@]}
echo "Total training jobs: ${TOTAL_JOBS}"

# Process jobs in parallel using available GPUs
JOB_INDEX=0
PIDS=()

for JOB in "${JOBS[@]}"; do
    # Parse job
    SEED=$(echo ${JOB} | cut -d':' -f1)
    ABLATION_TYPE=$(echo ${JOB} | cut -d':' -f2)

    # Select GPU (round-robin)
    GPU_INDEX=$((JOB_INDEX % ${#GPU_DEVICES[@]}))
    GPU_DEVICE=${GPU_DEVICES[$GPU_INDEX]}

    echo ""
    echo "=========================================="
    echo "Job $((JOB_INDEX + 1))/${TOTAL_JOBS}: Training ${ABLATION_TYPE} with seed ${SEED} on GPU ${GPU_DEVICE}"
    echo "=========================================="

    # Create seed directory if it doesn't exist
    SEED_MODEL_DIR="${BASE_OUTPUT_DIR}/seed${SEED}"
    mkdir -p -- "$SEED_MODEL_DIR"

    GROUND_TRUTH_PATH="data/ablation_study/seed${SEED}/[sft]ground_truth_final_gifarc_output_1613_${ABLATION_TYPE}_seed${SEED}_1000.jsonl"
    OUTPUT_DIR="${SEED_MODEL_DIR}/Mistral-NeMo-Minitron-8B-Base-GIFARC-ablation-${ABLATION_TYPE}-seed${SEED}"

    # Run training in background
    (
        CUDA_VISIBLE_DEVICES=${GPU_DEVICE} python "$SCRIPT_DIR/sft_train.py" \
            --model_name ${MODEL_NAME} \
            --ground_truth_data_path "$GROUND_TRUTH_PATH" \
            --num_epochs ${NUM_EPOCHS} \
            --max_length ${MAX_LENGTH} \
            --max_new_tokens ${MAX_NEW_TOKENS} \
            --output_dir "$OUTPUT_DIR" \
            --seed ${SEED}

        if [ $? -eq 0 ]; then
            echo "✓ Training ${ABLATION_TYPE} with seed ${SEED} on GPU ${GPU_DEVICE} completed successfully"
        else
            echo "✗ Training ${ABLATION_TYPE} with seed ${SEED} on GPU ${GPU_DEVICE} failed"
        fi
    ) &

    PIDS+=($!)
    JOB_INDEX=$((JOB_INDEX + 1))

    # Wait if we've started as many jobs as we have GPUs
    if [ $((JOB_INDEX % ${#GPU_DEVICES[@]})) -eq 0 ]; then
        echo ""
        echo "Waiting for current batch of ${#GPU_DEVICES[@]} jobs to complete..."
        for PID in "${PIDS[@]}"; do
            wait $PID
        done
        PIDS=()
        echo "Batch completed. Starting next batch..."
    fi
done

# Wait for any remaining jobs
if [ ${#PIDS[@]} -gt 0 ]; then
    echo ""
    echo "Waiting for final batch to complete..."
    for PID in "${PIDS[@]}"; do
        wait $PID
    done
fi

echo ""
echo "=========================================="
echo "All training completed!"
echo "=========================================="
