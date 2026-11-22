#!/usr/bin/env bash
# Waterfowl Workbench — the terminal interface for this project.
#
#   ./waterfowl_detection.sh        open the main menu
#   ./waterfowl_detection.sh jobs   open one screen directly (data, train, evaluate, detect,
#                                   review, export, jobs, settings), then the main menu
#
# Runs in the `waterfowl` conda environment.  WATERFOWL_PYTHON overrides which Python.
# Full manual: WATERFOWL_DETECTION_MANUAL.md
set -euo pipefail

here="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
cd "$here"

py="${WATERFOWL_PYTHON:-}"
if [[ -z "$py" ]]; then
  if [[ "${CONDA_DEFAULT_ENV:-}" == "waterfowl" ]] && command -v python >/dev/null; then
    py="$(command -v python)"
  else
    for base in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/mambaforge" \
                "/opt/conda" "${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}"; do
      if [[ -n "$base" && -x "$base/envs/waterfowl/bin/python" ]]; then
        py="$base/envs/waterfowl/bin/python"
        break
      fi
    done
  fi
fi
if [[ -z "$py" ]]; then
  py="$(command -v python3 || command -v python || true)"
  echo "note: the 'waterfowl' conda environment was not found; using ${py:-no python}." >&2
  echo "      Set WATERFOWL_PYTHON=/path/to/python to choose one." >&2
fi
if ! "$py" -c "import rich, prompt_toolkit" 2>/dev/null; then
  echo "The workbench needs 'rich' and 'prompt_toolkit' in $py:" >&2
  echo "  $py -m pip install rich prompt_toolkit" >&2
  exit 1
fi

exec "$py" -m workbench "$@"
