#!/usr/bin/env bash
set -euo pipefail
MODEL=${MODEL:-black-forest-labs/FLUX.1-dev}
OUTPUT=${OUTPUT:-outputs/drawbench/flux}
PYTHON=${PYTHON:-python}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
# Space-separated intervals; use identical generation arguments for run_base.sh.
INTERVALS=${INTERVALS:-2 3 4 5 6 7}
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if ! [[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
    echo "NPROC_PER_NODE must be a positive integer" >&2
    exit 1
fi
LAUNCH=("$PYTHON" -m torch.distributed.run --standalone --nnodes=1
    --nproc_per_node "$NPROC_PER_NODE" --module)
read -r -a VALUES <<< "$INTERVALS"
for interval in "${VALUES[@]}"; do
    RUN="$OUTPUT/interval_$interval"
    "${LAUNCH[@]}" eval.drawbench.generate --model "$MODEL" "$@" \
        --method fora --interval "$interval" --output "$RUN/images"
    "${LAUNCH[@]}" eval.drawbench.score --baseline "$OUTPUT/baseline" \
        --candidate "$RUN/images" --output "$RUN/scores"
    "$PYTHON" -m eval.drawbench.export --scores "$RUN/scores" --output "$RUN/report"
done
