#!/usr/bin/env bash
set -euo pipefail

BASE=${1:?usage: make_eval_checkpoint.sh BASE_CKPT ACTION_HEAD OUTPUT_DIR}
HEAD=${2:?usage: make_eval_checkpoint.sh BASE_CKPT ACTION_HEAD OUTPUT_DIR}
OUT=${3:?usage: make_eval_checkpoint.sh BASE_CKPT ACTION_HEAD OUTPUT_DIR}

test -d "$BASE"
test -f "$HEAD"
mkdir -p "$OUT"

# Link the immutable 15 GB base model instead of copying it.  Do not link an
# existing action head; the output must contain exactly one candidate head.
while IFS= read -r -d '' item; do
  name=$(basename "$item")
  case "$name" in
    action_head--*_checkpoint.pt) continue ;;
  esac
  ln -sfn "$item" "$OUT/$name"
done < <(find "$BASE" -mindepth 1 -maxdepth 1 -print0)

rm -f "$OUT"/action_head--*_checkpoint.pt
cp "$HEAD" "$OUT/$(basename "$HEAD")"

count=$(find "$OUT" -maxdepth 1 -type f -name 'action_head--*_checkpoint.pt' | wc -l)
test "$count" -eq 1
echo "evaluation checkpoint: $OUT"
find "$OUT" -maxdepth 1 -type f -name 'action_head--*_checkpoint.pt' -printf '%f\n'
