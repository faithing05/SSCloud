#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage:"
    echo "  $0 --input-dir DIR --output-dir DIR [--results-dir DIR --epoch N | --model PATH] [options]"
    echo
    echo "Required:"
    echo "  --input-dir DIR      Folder with input images"
    echo "  --output-dir DIR     Folder for output masks"
    echo
    echo "Checkpoint source (choose one):"
    echo "  --results-dir DIR    Training results directory containing checkpoints/"
    echo "  --epoch N            Epoch number to load from results-dir"
    echo "  --model PATH         Direct path to .pth model"
    echo
    echo "Options:"
    echo "  --scale F            Input scale factor (default: 0.1)"
    echo "  --classes N          Number of classes (default: 6)"
    echo "  --mask-threshold F   Binary threshold (default: 0.5)"
    echo "  --no-attention-maps  Disable saving attention maps"
    echo "  --no-transformer     Disable transformer bottleneck"
    echo "  --no-attention       Disable attention gates"
}

INPUT_DIR=""
OUTPUT_DIR=""
RESULTS_DIR=""
EPOCH=""
MODEL_PATH=""

SCALE=0.1
CLASSES=6
MASK_THRESHOLD=0.5
SAVE_ATTENTION_MAPS=1
USE_TRANSFORMER=1
USE_ATTENTION=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --epoch)
            EPOCH="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
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

if [[ -z "${INPUT_DIR}" || -z "${OUTPUT_DIR}" ]]; then
    echo "Error: --input-dir and --output-dir are required"
    usage
    exit 1
fi

if [[ -n "${MODEL_PATH}" ]]; then
    if [[ ! -f "${MODEL_PATH}" ]]; then
        echo "Error: model file not found: ${MODEL_PATH}"
        exit 1
    fi
else
    if [[ -z "${RESULTS_DIR}" || -z "${EPOCH}" ]]; then
        echo "Error: provide either --model PATH or both --results-dir DIR --epoch N"
        exit 1
    fi
fi

if [[ ! -d "${INPUT_DIR}" ]]; then
    echo "Error: input directory not found: ${INPUT_DIR}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

shopt -s nullglob
images=(
    "${INPUT_DIR}"/*.png
    "${INPUT_DIR}"/*.jpg
    "${INPUT_DIR}"/*.jpeg
    "${INPUT_DIR}"/*.bmp
    "${INPUT_DIR}"/*.tif
    "${INPUT_DIR}"/*.tiff
)

if [[ ${#images[@]} -eq 0 ]]; then
    echo "Error: no images found in ${INPUT_DIR}"
    exit 1
fi

echo "Found ${#images[@]} images. Starting batch inference..."

for image_path in "${images[@]}"; do
    image_name="$(basename "${image_path}")"
    image_stem="${image_name%.*}"
    output_mask="${OUTPUT_DIR}/${image_stem}_mask.png"

    cmd=(
        python predict.py
        -i "${image_path}"
        -o "${output_mask}"
        --scale "${SCALE}"
        --classes "${CLASSES}"
        --mask-threshold "${MASK_THRESHOLD}"
    )

    if [[ -n "${MODEL_PATH}" ]]; then
        cmd+=(--model "${MODEL_PATH}")
    else
        cmd+=(--results-dir "${RESULTS_DIR}" --epoch "${EPOCH}")
    fi

    if [[ "${USE_TRANSFORMER}" -eq 1 ]]; then
        cmd+=(--use-transformer)
    else
        cmd+=(--no-transformer)
    fi

    if [[ "${USE_ATTENTION}" -eq 1 ]]; then
        cmd+=(--use-attention)
    else
        cmd+=(--no-attention)
    fi

    if [[ "${SAVE_ATTENTION_MAPS}" -eq 1 ]]; then
        cmd+=(--save-attention)
    fi

    echo "Processing: ${image_name}"
    "${cmd[@]}"
done

echo "Batch inference finished. Outputs saved to: ${OUTPUT_DIR}"
