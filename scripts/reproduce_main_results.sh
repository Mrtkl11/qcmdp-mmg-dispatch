set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_PATH="${1:-${ROOT_DIR}/data/Environment_data_2018.csv}"
QCMDP_CONFIG="${2:-${ROOT_DIR}/configs/experiment_config.template.json}"
BASELINE_CONFIG="${3:-${ROOT_DIR}/configs/baselines.template.json}"

cd "${ROOT_DIR}"

python -m src.qcmdp_single_mg_training --data-path "${DATA_PATH}" --config-json "${QCMDP_CONFIG}" --mg-id 1 --output-dir "${ROOT_DIR}/results/single_mg/qcmdp"

for algorithm in a2c ppo qppo sac constrained_ppo; do
  python -m baselines.runner --algorithm "${algorithm}" --data-path "${DATA_PATH}" --config-json "${BASELINE_CONFIG}" --mg-id 1 --output-dir "${ROOT_DIR}/results/single_mg/${algorithm}"
done

python -m src.ring_federated_qcmdp_training --data-path "${DATA_PATH}" --config-json "${QCMDP_CONFIG}" --output-dir "${ROOT_DIR}/results/federated/qcmdp"

for algorithm in a2c ppo qppo sac constrained_ppo; do
  python -m baselines.federated --algorithm "${algorithm}" --data-path "${DATA_PATH}" --config-json "${BASELINE_CONFIG}" --output-dir "${ROOT_DIR}/results/federated/${algorithm}"
done
