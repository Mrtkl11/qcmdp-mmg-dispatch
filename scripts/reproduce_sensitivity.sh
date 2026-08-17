set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: bash scripts/reproduce_sensitivity.sh DATA_PATH CONFIG_JSON [CONFIG_JSON ...]"
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_PATH="$1"
shift

cd "${ROOT_DIR}"

for config_path in "$@"; do
  run_name="$(basename "${config_path}" .json)"
  python -m src.qcmdp_single_mg_training --data-path "${DATA_PATH}" --config-json "${config_path}" --mg-id 1 --output-dir "${ROOT_DIR}/results/sensitivity/${run_name}"
done
