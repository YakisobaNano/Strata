#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 BASE_STRATA OPT_STRATA [common strata args ...]" >&2
  echo "env: STRATA_BENCH_RUNS=5 STRATA_BENCH_OUT=v100-ab-YYYYmmdd-HHMMSS STRATA_BENCH_NSYS=0" >&2
  exit 2
fi

base=$1
opt=$2
shift 2

runs=${STRATA_BENCH_RUNS:-5}
out=${STRATA_BENCH_OUT:-v100-ab-$(date +%Y%m%d-%H%M%S)}
nsys_run=${STRATA_BENCH_NSYS:-0}
mkdir -p "$out"

if [[ ! -x "$base" || ! -x "$opt" ]]; then
  echo "both BASE_STRATA and OPT_STRATA must be executable" >&2
  exit 2
fi

{
  echo "date: $(date --iso-8601=seconds 2>/dev/null || date)"
  echo "host: $(hostname)"
  echo "base: $base"
  echo "opt:  $opt"
  echo "runs: $runs"
  echo "args:"
  printf '  %q' "$@"
  echo
  command -v nvidia-smi >/dev/null && nvidia-smi
} >"$out/environment.txt" 2>&1

run_one() {
  local label=$1
  local bin=$2
  local i log
  echo "== $label warmup =="
  "$bin" "$@" >"$out/$label-warmup.log" 2>&1

  for ((i=1; i<=runs; ++i)); do
    log="$out/$label-$(printf '%02d' "$i").log"
    echo "== $label run $i/$runs =="
    /usr/bin/time -f $'wall_s=%e\nuser_s=%U\nsys_s=%S\nmaxrss_kb=%M' \
      "$bin" "$@" >"$log" 2>&1
  done

  if [[ "$nsys_run" == "1" ]]; then
    if ! command -v nsys >/dev/null; then
      echo "STRATA_BENCH_NSYS=1 but nsys was not found" >&2
      exit 2
    fi
    echo "== $label nsys =="
    nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
      -o "$out/$label-nsys" "$bin" "$@" >"$out/$label-nsys.log" 2>&1
    nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum \
      "$out/$label-nsys.nsys-rep" >"$out/$label-nsys-stats.txt" 2>&1 || true
  fi
}

run_one base "$base" "$@"
run_one opt "$opt" "$@"

cat <<EOF
A/B logs written to: $out

Compare, in this order:
  1. identical prompt / seed / temperature / spec depth
  2. accepted drafts and tokens-per-step (must remain comparable)
  3. ms/step and MTP draft ms
  4. verify profile ms/window
  5. with STRATA_BENCH_NSYS=1: IQ2_S/native expert kernel time, launch count,
     cudaMemcpy/cudaMemcpyAsync time, and host synchronization/API time

Do not accept a tok/s-only win when acceptance or output differs.
EOF
