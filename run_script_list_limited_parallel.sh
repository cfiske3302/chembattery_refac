#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# run_script_list_limited_parallel.sh
# Runs every config listed in the supplied file concurrently but with an optional
# upper bound on the number of simultaneous jobs.
#
# Usage:
#   ./run_script_list_limited_parallel.sh list.txt [MAX_PARALLEL]
#
# Arguments:
#   list.txt      – File containing paths (one per line) to config files.
#   MAX_PARALLEL  – Optional positive integer. Limits how many training jobs
#                   run at once. Omit or set to 0 for unlimited parallelism.
#
# Behaviour:
#   • Validates inputs and list file existence.
#   • Skips blank lines or comments (# ...).
#   • Launches: python main.py --config "<path>" --train --eval
#     in a background subshell for each valid line.
#   • Enforces the maximum-parallel limit by throttling launches.
#   • Waits for all jobs to finish, printing individual success / failure.
# ------------------------------------------------------------------------------

if (( $# < 1 || $# > 2 )); then
  echo "Usage: $0 <file_with_script_paths> [max_parallel_jobs]"
  exit 1
fi

list_file="$1"
max_parallel="${2:-0}"        # 0 → unlimited
echo "Max parallel jobs: $max_parallel"
# Validate list file
if [[ ! -f "$list_file" ]]; then
  echo "Error: '$list_file' does not exist."
  exit 1
fi

# Validate max_parallel if provided
if ! [[ "$max_parallel" =~ ^[0-9]+$ ]]; then
  echo "Error: max_parallel_jobs must be a non-negative integer."
  exit 1
fi

# convert to integer
max_parallel=$((max_parallel))

declare -a pids=()
declare -A pid_to_script=()

while IFS= read -r script_path_raw || [[ -n $script_path_raw ]]; do
  # Trim whitespace
  script_path="$(echo "$script_path_raw" | xargs)"

  # Skip blanks and comments
  [[ -z $script_path || $script_path == \#* ]] && continue

  if [[ ! -f $script_path ]]; then
    echo "Warning: '$script_path' not found – skipping."
    continue
  fi

  # Throttle if limit set (>0)
  if (( max_parallel > 0 )); then
    while (( $(jobs -pr | wc -l) >= max_parallel )); do
      # All parallel slots are busy; wait a minute before re-checking
      sleep 60
    done
  fi

  echo "===== Starting: $script_path ====="
  (
    python main.py --config "$script_path" --train --eval
  ) &
  pid=$!
  pids+=("$pid")
  pid_to_script["$pid"]="$script_path"
  echo "    Launched PID $pid"
done < "$list_file"

echo "--------------------------------------------------------"
echo "All jobs launched. Waiting for completion..."
echo "--------------------------------------------------------"

# Final wait – iterate over pids and report status
for pid in "${pids[@]}"; do
  script="${pid_to_script[$pid]}"
  if wait "$pid"; then
    echo "✓ SUCCESS – $script (PID $pid)"
  else
    status=$?
    echo "✗ FAILURE – $script (PID $pid) exited with code $status"
  fi
done

echo "--------------------------------------------------------"
echo "All jobs finished."
