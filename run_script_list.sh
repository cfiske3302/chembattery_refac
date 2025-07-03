#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# run_script_list.sh
# Runs each script listed (one per line) in the supplied file, sequentially.
#
# Usage:
#   ./run_script_list.sh /path/to/script_list.txt
#
# Script list is a list of config files to run sequentially.
#
# Behaviour:
#   • Each valid script is executed in the order it appears.
#   • If a listed path is missing, a warning is printed and processing continues.
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

while IFS= read -r script_path || [[ -n $script_path ]]; do
  # Trim surrounding whitespace
  script_path="$(echo "$script_path" | xargs)"

  # Skip blank lines and comment lines
  [[ -z $script_path || $script_path == \#* ]] && continue

  if [[ ! -f $script_path ]]; then
    echo "Warning: '$script_path' not found – skipping."
    continue
  fi

  echo "===== Running: $script_path ====="
  if [[ -f $script_path ]]; then
    python main.py --config "$script_path" --train --eval
  fi
  echo "===== Finished: $script_path ====="
done < "$list_file"
