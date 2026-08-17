set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_PATH="${1:-${ROOT_DIR}/data/Environment_data_2018.csv}"
QCMDP_CONFIG="${2:-${ROOT_DIR}/configs/experiment_config.template.json}"

cd "${ROOT_DIR}"

python -m src.ring_federated_qcmdp_training --data-path "${DATA_PATH}" --config-json "${QCMDP_CONFIG}" --federated-config-json "${ROOT_DIR}/configs/fault_tolerance.paper.json" --output-dir "${ROOT_DIR}/results/ablation/topology_reconfiguration"

python -m src.ring_federated_qcmdp_training --data-path "${DATA_PATH}" --config-json "${QCMDP_CONFIG}" --federated-config-json "${ROOT_DIR}/configs/fault_tolerance_ablation.paper.json" --output-dir "${ROOT_DIR}/results/ablation/no_topology_reconfiguration"
