#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# run_script_list_parallel.sh
# Runs every script (one per line) found in the supplied list file *concurrently*,
# spawning each in its own subshell.
#
# Usage:
#   ./run_script_list_parallel.sh /path/to/script_list.txt
#
# List file:
#   Lines are interpreted as config-file paths for main.py.
#   • Blank lines and lines beginning with ‘#’ are ignored.
#   • Missing paths are warned and skipped.
#
# Behaviour:
#   • Each valid entry launches:
#       python main.py --config "<path>" --train --eval
#     in a background subshell, capturing its PID.
#   • All jobs run concurrently.
#   • After launching, the script waits for every PID, reporting individual
#     success / failure status without aborting the whole run if one fails.
# ------------------------------------------------------------------------------

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <file_with_script_paths>"
  exit 1
fi

list_file="$1"

if [[ ! -f "$list_file" ]]; then
  echo "Error: '$list_file' does not exist."
  exit 1
fi

# Arrays for tracking PIDs and mapping them back to their script paths
declare -a pids=()
declare -A pid_to_script=()

while IFS= read -r script_path_raw || [[ -n $script_path_raw ]]; do
  # Trim surrounding whitespace
  script_path="$(echo "$script_path_raw" | xargs)"

  # Skip blank lines and comment lines
  [[ -z $script_path || $script_path == \#* ]] && continue

  if [[ ! -f $script_path ]]; then
    echo "Warning: '$script_path' not found – skipping."
    continue
  fi

  echo "===== Starting: $script_path ====="
  (
    # Run in a subshell to guarantee isolated environment
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

# Wait for all jobs and report individual status
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
