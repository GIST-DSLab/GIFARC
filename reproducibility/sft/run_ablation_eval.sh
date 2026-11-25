#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# GPU device to use
GPU_DEVICE=0

# Seeds to iterate over
SEEDS=(444 555 666)

# Base configuration
MAX_LENGTH=6144
MAX_NEW_TOKENS=2048

# Root of the existing checkpoint layout.
BASE_STORAGE_PATH="${GIFARC_CHECKPOINT_ROOT:-./models}"

# Models to evaluate
MODELS=("qwen3_14b")
# MODELS=("mistral_8b" "qwen3_14b")  # Uncomment to include Qwen3-14B

# Model configurations
declare -A MODEL_NAMES
declare -A MODEL_PREFIXES

# Mistral-NeMo-Minitron-8B configuration
MODEL_NAMES["mistral_8b"]="nvidia/Mistral-NeMo-Minitron-8B-Base"
MODEL_PREFIXES["mistral_8b"]="Mistral-8B"

# Qwen3-14B configuration
MODEL_NAMES["qwen3_14b"]="Qwen/Qwen3-14B-Base"
MODEL_PREFIXES["qwen3_14b"]="Qwen3-14B"

# Training source configurations
# Each training source has: base_folder, sft_prefix, ablation_type, output_label
# Format: "base_folder|sft_prefix|ablation_type|output_label"
declare -A TRAIN_SRC_GIFARC_BASE_FOLDERS
declare -A TRAIN_SRC_GIFARC_SFT_PREFIXES
declare -A TRAIN_SRC_ARC_AGI2_BASE_FOLDERS
declare -A TRAIN_SRC_ARC_AGI2_SFT_PREFIXES

# GIFARC SFT checkpoint paths
TRAIN_SRC_GIFARC_BASE_FOLDERS["mistral_8b"]="Mistral-NeMo-Minitron-8B-GIFARC-ablation"
TRAIN_SRC_GIFARC_SFT_PREFIXES["mistral_8b"]="Mistral-NeMo-Minitron-8B-Base-GIFARC-ablation"
TRAIN_SRC_GIFARC_BASE_FOLDERS["qwen3_14b"]="Qwen-3-14B-GIFARC-ablation"
TRAIN_SRC_GIFARC_SFT_PREFIXES["qwen3_14b"]="Qwen3-14B-GIFARC-ablation"

# ARC-AGI-2 SFT checkpoint paths
TRAIN_SRC_ARC_AGI2_BASE_FOLDERS["mistral_8b"]="Mistral-NeMo-Minitron-8B-ARC-AGI2-train"
TRAIN_SRC_ARC_AGI2_SFT_PREFIXES["mistral_8b"]="Mistral-NeMo-Minitron-8B-Base-ARC-AGI2-train"
TRAIN_SRC_ARC_AGI2_BASE_FOLDERS["qwen3_14b"]="Qwen-3-14B-ARC-AGI2-train"
TRAIN_SRC_ARC_AGI2_SFT_PREFIXES["qwen3_14b"]="Qwen3-14B-ARC-AGI2-train"

# Training sources to evaluate
# Format: "source_name:ablation_type"
TRAINING_SOURCES=("gifarc:full" "arc_agi2:base")

# Model types to evaluate (base = original model, sft = fine-tuned model)
MODEL_TYPES=("base" "sft")

# Evaluation datasets (ARC-AGI-2 only for this comparison experiment)
DATASETS=("arc_agi2")

# Dataset configurations
declare -A DATASET_INPUT_PATHS
declare -A DATASET_GT_PATHS
declare -A DATASET_SAMPLE_COUNTS

DATASET_INPUT_PATHS["arc_agi1"]="data/arc_agi1_base_eval_all_tests/input_arc_agi1_eval_arc_419.jsonl"
DATASET_GT_PATHS["arc_agi1"]="data/arc_agi1_base_eval_all_tests/ground_truth_arc_agi1_eval_arc_419.jsonl"
DATASET_SAMPLE_COUNTS["arc_agi1"]="419"

DATASET_INPUT_PATHS["arc_agi2"]="data/arc_agi2_base_eval_all_tests/input_arc_agi2_eval_arc_167.jsonl"
DATASET_GT_PATHS["arc_agi2"]="data/arc_agi2_base_eval_all_tests/ground_truth_arc_agi2_eval_arc_167.jsonl"
DATASET_SAMPLE_COUNTS["arc_agi2"]="167"

EVAL_MODE="transduction"
TEMPERATURE=0.0

echo "Starting ablation study evaluation with multiple seeds on GPU ${GPU_DEVICE}"
echo "=================================================="

# Helper function to get checkpoint path for a given model, training source, and seed
get_checkpoint_path() {
    local model=$1
    local train_src=$2
    local ablation_type=$3
    local seed=$4

    if [ "${train_src}" == "gifarc" ]; then
        local base_folder="${TRAIN_SRC_GIFARC_BASE_FOLDERS[$model]}"
        local sft_prefix="${TRAIN_SRC_GIFARC_SFT_PREFIXES[$model]}"
    elif [ "${train_src}" == "arc_agi2" ]; then
        local base_folder="${TRAIN_SRC_ARC_AGI2_BASE_FOLDERS[$model]}"
        local sft_prefix="${TRAIN_SRC_ARC_AGI2_SFT_PREFIXES[$model]}"
    fi

    echo "${BASE_STORAGE_PATH}/${base_folder}/${sft_prefix}-${ablation_type}-seed${seed}/checkpoint-500"
}

# Count total jobs
# base model: models * datasets * seeds (base model evaluated once per model, not per training source)
# sft models: models * training_sources * datasets * seeds
BASE_JOBS=$((${#MODELS[@]} * ${#DATASETS[@]} * ${#SEEDS[@]}))
SFT_JOBS=$((${#MODELS[@]} * ${#TRAINING_SOURCES[@]} * ${#DATASETS[@]} * ${#SEEDS[@]}))
TOTAL_JOBS=$((BASE_JOBS + SFT_JOBS))
JOB_COUNT=0

echo "Total jobs to run: ${TOTAL_JOBS}"
echo "  - Models: ${#MODELS[@]} (${MODELS[*]})"
echo "  - Datasets: ${#DATASETS[@]} (${DATASETS[*]})"
echo "  - Seeds: ${#SEEDS[@]} (${SEEDS[*]})"
echo "  - Training sources: ${#TRAINING_SOURCES[@]} (${TRAINING_SOURCES[*]})"
echo "  - Base model jobs: ${BASE_JOBS}"
echo "  - SFT model jobs: ${SFT_JOBS}"

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Evaluating model: ${MODEL}"
    echo "=========================================="

    MODEL_NAME="${MODEL_NAMES[$MODEL]}"
    MODEL_PREFIX="${MODEL_PREFIXES[$MODEL]}"

    for DATASET in "${DATASETS[@]}"; do
        echo ""
        echo "----------------------------------------"
        echo "Dataset: ${DATASET}"
        echo "----------------------------------------"

        INPUT_DATA_PATH="${DATASET_INPUT_PATHS[$DATASET]}"
        GROUND_TRUTH_EVAL_PATH="${DATASET_GT_PATHS[$DATASET]}"
        SAMPLE_COUNT="${DATASET_SAMPLE_COUNTS[$DATASET]}"

        for SEED in "${SEEDS[@]}"; do
            # Create results directory for this seed
            SEED_RESULTS_DIR="./results/seed${SEED}"
            mkdir -p "${SEED_RESULTS_DIR}" || exit "$?"

            # ---- Base model evaluation (no fine-tuning) ----
            JOB_COUNT=$((JOB_COUNT + 1))
            echo ""
            echo "Job ${JOB_COUNT}/${TOTAL_JOBS}: [${MODEL}] [${DATASET}] base seed ${SEED}..."

            EVAL_MODEL_PATH="${MODEL_NAME}"
            # Use naming compatible with existing base model results
            OUTPUT_PATH="${SEED_RESULTS_DIR}/${MODEL_PREFIX}-GIFARC-ablation-base-seed${SEED}_${DATASET}-eval-${SAMPLE_COUNT}.jsonl"

            if CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" python "$SCRIPT_DIR/sft_eval.py" \
                --model_name "${MODEL_NAME}" \
                --model_path "${EVAL_MODEL_PATH}" \
                --input_data_path "${INPUT_DATA_PATH}" \
                --ground_truth_data_path "${GROUND_TRUTH_EVAL_PATH}" \
                --output_path "${OUTPUT_PATH}" \
                --max_length "${MAX_LENGTH}" \
                --max_new_tokens "${MAX_NEW_TOKENS}" \
                --evaluation_mode "${EVAL_MODE}" \
                --temperature "${TEMPERATURE}" \
                --seed "${SEED}"; then
                echo "✓ [${MODEL}] [${DATASET}] base seed ${SEED} completed"
            else
                STATUS=$?
                echo "✗ [${MODEL}] [${DATASET}] base seed ${SEED} failed" >&2
                exit "${STATUS}"
            fi

            # ---- SFT model evaluations (per training source) ----
            for TRAIN_SRC_SPEC in "${TRAINING_SOURCES[@]}"; do
                # Parse "source_name:ablation_type"
                TRAIN_SRC="${TRAIN_SRC_SPEC%%:*}"
                ABLATION_TYPE="${TRAIN_SRC_SPEC##*:}"

                JOB_COUNT=$((JOB_COUNT + 1))
                echo ""
                echo "Job ${JOB_COUNT}/${TOTAL_JOBS}: [${MODEL}] [${DATASET}] sft-${TRAIN_SRC} ${ABLATION_TYPE} seed ${SEED}..."

                EVAL_MODEL_PATH=$(get_checkpoint_path "${MODEL}" "${TRAIN_SRC}" "${ABLATION_TYPE}" "${SEED}")

                # Naming convention:
                #   GIFARC:   {MODEL_PREFIX}-GIFARC-ablation-sft-{ablation}-seed{N}_{dataset}-eval-{count}.jsonl  (compatible with existing results)
                #   ARC-AGI-2: {MODEL_PREFIX}-ARC-AGI2-train-sft-{ablation}-seed{N}_{dataset}-eval-{count}.jsonl
                if [ "${TRAIN_SRC}" == "gifarc" ]; then
                    OUTPUT_PATH="${SEED_RESULTS_DIR}/${MODEL_PREFIX}-GIFARC-ablation-sft-${ABLATION_TYPE}-seed${SEED}_${DATASET}-eval-${SAMPLE_COUNT}.jsonl"
                elif [ "${TRAIN_SRC}" == "arc_agi2" ]; then
                    OUTPUT_PATH="${SEED_RESULTS_DIR}/${MODEL_PREFIX}-ARC-AGI2-train-sft-${ABLATION_TYPE}-seed${SEED}_${DATASET}-eval-${SAMPLE_COUNT}.jsonl"
                fi

                # Check if model checkpoint exists
                if [ ! -d "${EVAL_MODEL_PATH}" ]; then
                    echo "Error: Model checkpoint not found at ${EVAL_MODEL_PATH}" >&2
                    exit 1
                fi

                if CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" python "$SCRIPT_DIR/sft_eval.py" \
                    --model_name "${MODEL_NAME}" \
                    --model_path "${EVAL_MODEL_PATH}" \
                    --input_data_path "${INPUT_DATA_PATH}" \
                    --ground_truth_data_path "${GROUND_TRUTH_EVAL_PATH}" \
                    --output_path "${OUTPUT_PATH}" \
                    --max_length "${MAX_LENGTH}" \
                    --max_new_tokens "${MAX_NEW_TOKENS}" \
                    --evaluation_mode "${EVAL_MODE}" \
                    --temperature "${TEMPERATURE}" \
                    --seed "${SEED}"; then
                    echo "✓ [${MODEL}] [${DATASET}] sft-${TRAIN_SRC} ${ABLATION_TYPE} seed ${SEED} completed"
                else
                    STATUS=$?
                    echo "✗ [${MODEL}] [${DATASET}] sft-${TRAIN_SRC} ${ABLATION_TYPE} seed ${SEED} failed" >&2
                    exit "${STATUS}"
                fi
            done
        done
    done
done

echo ""
echo "=========================================="
echo "All evaluation completed!"
echo "=========================================="
