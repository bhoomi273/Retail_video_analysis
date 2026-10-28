#!/bin/bash
set -euo pipefail

mkdir -p /app/outputs

task="${1:-all}"

case "$task" in
  all)
    python src/process_entrance.py --video data/entrance.mp4 --skip 1
    python src/process_interior.py --video data/interior.mp4 --skip 1
    ;;
  entrance)
    python src/process_entrance.py --video data/entrance.mp4 --skip 1
    ;;
  interior)
    python src/process_interior.py --video data/interior.mp4 --skip 1
    ;;
  *)
    echo "Usage: docker run <image> [all|entrance|interior]" >&2
    exit 1
    ;;
esac
