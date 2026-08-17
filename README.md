# Anonymous Code Submission for Safe Multi-Microgrid Dispatch

This repository contains the code supplement for an anonymous manuscript under double-blind review.

It provides the proposed Q-CMDP method, the IEEE 33-bus multi-microgrid environment, fault-tolerant ring federation, QPPO, A2C, PPO, SAC, and constrained PPO.

## Method

Q-CMDP combines a classical Gaussian actor, separate variational quantum reward and safety critics, a primal-dual CMDP update, and decentralized ring federation. Each VQC applies one RX/RY/RZ state-encoding stage, repeated RY/RZ variational layers with nearest-neighbor CZ entanglement, Pauli-Z measurements, and a single linear value readout. The environment applies equipment, SOC, fuel, PCC, and network limits, but it does not use posterior QP action projection, CVXPY, OSQP, or another optimization safety layer.

The reward channel contains economic dispatch cost, recourse cost, network loss, and training regularizers. The CMDP cost channel contains voltage excess, dispatch-channel capacity excess, and branch-flow capacity excess.

Each active client trains from the same incoming global model. Local parameters are combined with an equal-weight arithmetic mean, VQC angles with an equal-weight circular mean, and the result with the previous global state using the configured inter-round momentum. Unavailable clients are bypassed when topology reconfiguration is enabled.

## State and action spaces

The normalized state follows the paper ordering:

```text
[PV, wind, load, SOC, diesel output, remaining fuel, PCC exchange,
 33 bus voltages, 32 branch loading states, network loss, buy price, sell price]
```

The resulting observation dimension is 75. The continuous policy action is:

```text
[diesel request, battery request, PCC exchange request]
```

Each action component lies in `[-1, 1]`. The environment maps requests to physically bounded schedules and represents residual deficits or surpluses through load shedding or renewable curtailment. Network violations remain observable CMDP costs rather than post-processed action corrections.

## Repository layout

```text
qcmdp-upload/
|-- baselines/
|-- configs/
|-- data/
|-- experiments/
|-- hardware/
|-- results/
|-- scripts/
|-- src/
|-- CITATION.cff
|-- CONTRIBUTING.md
|-- LICENSE
|-- README.md
|-- environment.yml
|-- pyproject.toml
`-- requirements.txt
```

The main implementation is located in:

- `src/ieee33_mmg_env.py`: IEEE 33-bus multi-microgrid environment and DistFlow evaluation;
- `src/qcmdp_model_pennylane.py`: classical actor and VQC critic;
- `src/qcmdp_single_mg_training.py`: primal-dual Q-CMDP training and evaluation;
- `src/ring_federated_qcmdp_training.py`: fault-tolerant equal-weight ring federation.

The comparison implementations are located in `baselines/`. All methods use the same environment, state/action definitions, cost interface, evaluation horizon, and federated aggregation rule.

The `hardware/` utilities export the trained dual-VQC circuits without credentials and summarize repeated QPU Pauli-Z measurements after authorized cloud execution.

## Configuration

Optimizer settings, architecture choices, rollout lengths, batch sizes, update counts, CMDP settings, and federation settings are stored only in JSON files under `configs/`. Python source files do not contain experiment-level training defaults.

- `environment.paper.json`: physical system, reward, and safety-cost definition;
- `experiment_config.template.json`: Q-CMDP and standard federation settings;
- `baselines.template.json`: all comparison-method settings;
- `fault_tolerance.paper.json`: MG3 outage during rounds 21-40 with bypass;
- `fault_tolerance_ablation.paper.json`: the same outage without topology reconstruction.
- `hardware_evaluation.template.json`: QPU payload and repeated-measurement paths.

The supplied JSON values follow the paper table where reported and preserve necessary implementation settings not listed in the table. A resolved configuration is saved with each run.

## Installation

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
```

Alternatively:

```bash
conda env create -f environment.yml
conda activate qcmdp-mmg
```

## Dataset

The local experiment CSV uses the columns:

```text
time, household_power, solar_power, wind_power, EUR/kWh
```

The merged third-party dataset is excluded from Git by default. Obtain the source series under their applicable terms, create the merged CSV, and pass it with `--data-path`. The repository includes only a small synthetic test fixture, which is not intended for paper-result reproduction.

## Single-microgrid experiments

```bash
python -m src.qcmdp_single_mg_training --data-path data/Environment_data_2018.csv --config-json configs/experiment_config.template.json --mg-id 1
python -m baselines.a2c --data-path data/Environment_data_2018.csv --config-json configs/baselines.template.json --mg-id 1
python -m baselines.ppo --data-path data/Environment_data_2018.csv --config-json configs/baselines.template.json --mg-id 1
python -m baselines.qppo --data-path data/Environment_data_2018.csv --config-json configs/baselines.template.json --mg-id 1
python -m baselines.sac --data-path data/Environment_data_2018.csv --config-json configs/baselines.template.json --mg-id 1
python -m baselines.constrained_ppo --data-path data/Environment_data_2018.csv --config-json configs/baselines.template.json --mg-id 1
```

QPPO uses the same classical actor and reward VQC architecture as Q-CMDP with PPO clipping, but it has no safety critic or Lagrange multiplier. C-PPO uses classical reward and safety critics with the same primal-dual constraint objective.

## Federated experiments

```bash
python -m src.ring_federated_qcmdp_training --data-path data/Environment_data_2018.csv --config-json configs/experiment_config.template.json
python -m baselines.federated --algorithm qppo --data-path data/Environment_data_2018.csv --config-json configs/baselines.template.json
```

Replace `qppo` with `a2c`, `ppo`, `sac`, or `constrained_ppo` for the other baselines.

## Reproduction scripts

Windows PowerShell:

```powershell
.\scripts\reproduce_main_results.ps1
.\scripts\reproduce_ablation.ps1
```

Bash:

```bash
bash scripts/reproduce_main_results.sh
bash scripts/reproduce_ablation.sh
```

The sensitivity runner accepts the dataset followed by one or more complete Q-CMDP JSON configurations:

```bash
bash scripts/reproduce_sensitivity.sh data/Environment_data_2018.csv configs/run_a.json configs/run_b.json
```

## Quantum hardware evaluation

```bash
python -m hardware.export_qpu_payload --config-json configs/hardware_evaluation.template.json
python -m hardware.summarize_qpu_results --config-json configs/hardware_evaluation.template.json
```

The export contains the exact paper-aligned VQC gates and trained parameters. QPU submission uses the researcher's authorized vendor account; credentials and account-specific submission code are not stored in this repository.

## Outputs

Runs write model checkpoints, resolved configurations, convergence tables, dispatch histories, bus-voltage and line-loading time series, and evaluation figures under `results/`. Existing experimental results are not modified by source preparation.

## Anonymous review

This code supplement is provided solely for anonymous peer review. Citation and author information will be restored after the review process.

## License

This project is released under the MIT License.
