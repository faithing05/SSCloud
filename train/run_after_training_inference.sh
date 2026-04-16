#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage:"
    echo "  $0 --results-dir DIR --input-dir DIR --output-dir DIR [options]"
    echo
    echo "Runs batch inference using a checkpoint from completed training."
    echo "By default it auto-selects the latest epoch checkpoint."
    echo
    echo "Required:"
    echo "  --results-dir DIR    Training results directory (must contain checkpoints/)"
    echo "  --input-dir DIR      Folder with input images"
    echo "  --output-dir DIR     Folder for output masks"
    echo
    echo "Options:"
    echo "  --epoch N            Use specific epoch checkpoint"
    echo "  --scale F            Input scale factor (default: 0.1)"
    echo "  --classes N          Number of classes (default: 9)"
    echo "  --mask-threshold F   Binary threshold (default: 0.5)"
    echo "  --no-attention-maps  Disable saving attention maps"
    echo "  --no-transformer     Disable transformer bottleneck"
    echo "  --no-attention       Disable attention gates"
    echo "  --no-collage         Disable collage generation"
    echo "  --collage-dir DIR    Folder for collages (default: OUTPUT_DIR/collages)"
}

RESULTS_DIR=""
INPUT_DIR=""
OUTPUT_DIR=""
EPOCH=""

SCALE=0.1
CLASSES=9
MASK_THRESHOLD=0.5
SAVE_ATTENTION_MAPS=1
USE_TRANSFORMER=1
USE_ATTENTION=1
GENERATE_COLLAGE=1
COLLAGE_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --epoch)
            EPOCH="$2"
            shift 2
            ;;
        --scale)
            SCALE="$2"
            shift 2
            ;;
        --classes)
            CLASSES="$2"
            shift 2
            ;;
        --mask-threshold)
            MASK_THRESHOLD="$2"
            shift 2
            ;;
        --no-attention-maps)
            SAVE_ATTENTION_MAPS=0
            shift
            ;;
        --no-transformer)
            USE_TRANSFORMER=0
            shift
            ;;
        --no-attention)
            USE_ATTENTION=0
            shift
            ;;
        --no-collage)
            GENERATE_COLLAGE=0
            shift
            ;;
        --collage-dir)
            COLLAGE_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "${RESULTS_DIR}" || -z "${INPUT_DIR}" || -z "${OUTPUT_DIR}" ]]; then
    echo "Error: --results-dir, --input-dir and --output-dir are required"
    usage
    exit 1
fi

CHECKPOINTS_DIR="${RESULTS_DIR}/checkpoints"
if [[ ! -d "${CHECKPOINTS_DIR}" ]]; then
    echo "Error: checkpoints directory not found: ${CHECKPOINTS_DIR}"
    exit 1
fi

if [[ -z "${EPOCH}" ]]; then
    shopt -s nullglob
    checkpoint_files=("${CHECKPOINTS_DIR}"/checkpoint_epoch*.pth)

    if [[ ${#checkpoint_files[@]} -eq 0 ]]; then
        echo "Error: no checkpoint files found in ${CHECKPOINTS_DIR}"
        exit 1
    fi

    max_epoch=-1
    for checkpoint_file in "${checkpoint_files[@]}"; do
        checkpoint_name="$(basename "${checkpoint_file}")"
        if [[ "${checkpoint_name}" =~ checkpoint_epoch([0-9]+)\.pth$ ]]; then
            epoch_num="${BASH_REMATCH[1]}"
            if (( epoch_num > max_epoch )); then
                max_epoch=${epoch_num}
            fi
        fi
    done

    if (( max_epoch < 0 )); then
        echo "Error: failed to parse epoch numbers from checkpoint filenames"
        exit 1
    fi

    EPOCH="${max_epoch}"
    echo "Auto-selected latest epoch checkpoint: ${EPOCH}"
fi

cmd=(
    bash "${SCRIPT_DIR}/run_batch_inference.sh"
    --input-dir "${INPUT_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --results-dir "${RESULTS_DIR}"
    --epoch "${EPOCH}"
    --scale "${SCALE}"
    --classes "${CLASSES}"
    --mask-threshold "${MASK_THRESHOLD}"
)

if [[ "${SAVE_ATTENTION_MAPS}" -eq 0 ]]; then
    cmd+=(--no-attention-maps)
fi

if [[ "${USE_TRANSFORMER}" -eq 0 ]]; then
    cmd+=(--no-transformer)
fi

if [[ "${USE_ATTENTION}" -eq 0 ]]; then
    cmd+=(--no-attention)
fi

"${cmd[@]}"

if [[ "${GENERATE_COLLAGE}" -eq 1 ]]; then
    if [[ -z "${COLLAGE_DIR}" ]]; then
        COLLAGE_DIR="${OUTPUT_DIR}/collages"
    fi

    python "${SCRIPT_DIR}/build_inference_collages.py" \
        --input-dir "${INPUT_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --collage-dir "${COLLAGE_DIR}"
fi
