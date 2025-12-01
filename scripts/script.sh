#!/usr/bin/env bash
set -euo pipefail

# Sync all graph folders from algorithms/dinics to fordflurkson and push_relabel
# Usage:
#   ./scripts/sync_graphs_from_dinics.sh        # perform copy
#   DRY_RUN=1 ./scripts/sync_graphs_from_dinics.sh   # show actions without copying

DRY_RUN_VAR=${DRY_RUN:-0}

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DINICS_DIR="$ROOT_DIR/algorithms/dinics"
TARGETS=("$ROOT_DIR/algorithms/fordflurkson" "$ROOT_DIR/algorithms/push_relabel")

if [ ! -d "$DINICS_DIR" ]; then
  echo "ERROR: Dinic folder not found: $DINICS_DIR" >&2
  exit 2
fi

echo "Will sync graph folders from: $DINICS_DIR"

# Find top-level subdirectories under algorithms/dinics that contain 'graph' in the name
# Explicit list of folders to copy from dinics
GRAPH_FOLDERS=(
  "general_v" "general_e" "unitcap_v" "unitcap_e"
  "simple_v" "simple_e" "worst_v" "worst_e" "graphs"
)

echo "Copying folders: ${GRAPH_FOLDERS[*]}"

for g in "${GRAPH_FOLDERS[@]}"; do
  src="$DINICS_DIR/$g"
  if [ ! -d "$src" ]; then
    echo "WARN: source folder not found: $src — skipping"
    continue
  fi

  # Copy graphs into both targets (fordflurkson and push_relabel)
  for tgt_base in "${TARGETS[@]}"; do
    tgt_dir="$tgt_base/$g"
    if [ "$DRY_RUN_VAR" -eq 1 ]; then
      echo "DRY-RUN: would create $tgt_dir and copy contents from $src/ -> $tgt_dir/"
    else
      echo "Copying $src/ -> $tgt_dir/"
      mkdir -p "$tgt_dir"
      if command -v rsync >/dev/null 2>&1; then
        rsync -av --delete "$src/" "$tgt_dir/"
      else
        cp -a "$src/". "$tgt_dir/"
      fi
    fi
  done
done

echo "Sync complete."
