# Decentralized Safety-Aware Quantum Federated Multi-Agent Reinforcement Learning for Multi-Microgrid Dispatch

This repository contains the public research code for the paper:

**Decentralized Safety-Aware Quantum Federated Multi-Agent Reinforcement Learning for Multi-Microgrid Dispatch**

The release is organized to make the method design, algorithmic structure, and implementation logic clear and reusable, while keeping the full reproduction workflow outside the public repository.

## Overview

The code follows the paper formulation:

- **Q-CMDP** for safety-aware reinforcement learning
- **Classical Gaussian actor**
- **Dual VQC critics**
- **Adaptive Lagrange multiplier**
- **Ring-based decentralized federated aggregation**
- **Offline-node bypass during aggregation**

## Repository Structure

```text
project/
|-- README.md
|-- environment.yml
|-- requirements.txt
|-- data/
|   `-- Environment_data_2018.csv
|-- configs/
|   `-- experiment_config.template.json
|-- src/
|   |-- ieee33_mmg_env.py
|   |-- qcmdp_model_pennylane.py
|   |-- qcmdp_single_mg_training.py
|   |-- release_config.py
|   `-- ring_federated_qcmdp_training.py
|-- baselines/
|-- experiments/
|-- scripts/
|   |-- reproduce_main_results.sh
|   |-- reproduce_ablation.sh
|   `-- reproduce_sensitivity.sh
|-- results/
`-- LICENSE
```

## File Description

- `src/ieee33_mmg_env.py`
  Public IEEE 33-bus multi-microgrid environment implementation with:
  - distributed energy resource modeling,
  - battery and diesel dynamics,
  - grid interaction,
  - simplified network power flow,
  - economic reward,
  - safety constraint cost.

- `src/qcmdp_model_pennylane.py`
  Model definitions for:
  - `GaussianActor`
  - `VQCCritic`
  - `QuantumCriticConfig`

- `src/qcmdp_single_mg_training.py`
  Single-microgrid Q-CMDP training and evaluation entrypoint.

- `src/ring_federated_qcmdp_training.py`
  Ring-federated multi-microgrid Q-CMDP training entrypoint.

- `src/release_config.py`
  Public-release utilities for:
  - loading external dataset paths,
  - loading optional JSON experiment overrides,
  - merging configuration overrides.

- `configs/experiment_config.template.json`
  Public configuration template for training. This is a template rather than a paper-final reproduction file.

## Method Mapping

The code is aligned with the paper in the following way:

1. **Local safe reinforcement learning**
   Each microgrid is modeled as a constrained Markov decision process. The actor maximizes the economic objective, while the safety critic estimates constraint cost. The Lagrange multiplier is updated online.

2. **Hybrid quantum-classical design**
   The actor is classical for continuous dispatch control. The reward critic and safety critic are implemented as variational quantum circuit critics.

3. **Ring federated learning**
   Federated training is implemented as sequential ring aggregation over active clients. If a node is inactive, it is bypassed during aggregation.

## State, Action, Reward, and Constraint

- Action definition:
  `a_t = [P_t^{dg}, P_t^{bat}, P_t^{grid}]`

- Environment outputs:
  - `reward`: economic objective
  - `constraint_cost`: safety-related violation cost

- Observation includes information related to:
  - renewable generation,
  - load demand,
  - battery SOC,
  - diesel output,
  - fuel level,
  - grid exchange,
  - voltage and line-related network features,
  - electricity prices,
  - short-horizon next-step signals,
  - previous action.

## Requirements

Recommended Python version:

- `Python 3.10+`

Install dependencies:

```bash
pip install numpy pandas matplotlib torch pennylane gymnasium
```

Alternatively, create the reproducible Conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate qcmdp-mmg
```

If your environment already has `gym`, the code can also use it directly. Otherwise it falls back to `gymnasium`.

## Public Release Note

This public repository is intended to expose the **algorithmic pipeline** and **implementation logic** of the framework.

It does **not** include the full reproduction-grade dataset and exact final experiment setup used for paper-ready result generation.

That means:

- you must provide your own external dataset path,
- you may need your own aligned preprocessing workflow,
- the included JSON config is a public template rather than the exact final paper configuration.

## Data Input

Training requires an external CSV dataset.

You must provide the dataset through one of the following:

- command-line argument `--data-path`
- environment variable `QCMDP_DATA_PATH`

If neither is provided, training will stop with a clear error message.

## Running the Code

### 1. Single-Microgrid Training

```bash
python src/qcmdp_single_mg_training.py --data-path data/Environment_data_2018.csv
```

Optional configuration override:

```bash
python src/qcmdp_single_mg_training.py --data-path data/Environment_data_2018.csv --config-json configs/experiment_config.template.json
```

This script will:

- train a single-microgrid Q-CMDP agent,
- save actor and critic weights,
- export convergence data,
- generate evaluation figures and CSV outputs.

### 2. Ring-Federated Multi-Microgrid Training

```bash
python src/ring_federated_qcmdp_training.py --data-path data/Environment_data_2018.csv
```

This script will:

- train all microgrids in a federated manner,
- perform ring aggregation across active clients,
- save global checkpoints,
- export reward and constraint-cost convergence curves,
- generate final evaluation plots for each microgrid.

## Configurable Environment Variables

For federated training, the following environment variables are supported:

- `FL_ROUNDS`
  Number of federated rounds.

- `LOCAL_UPDATES`
  Number of local updates per client in each federated round.

- `ACTIVE_INDICATORS`
  Active client mask, for example:
  `1,1,0,1,1`

- `RING_ALPHA`
  Ring aggregation mixing coefficient.

- `FED_SEED`
  Random seed for federated training.

Example on Windows CMD:

```bash
set FL_ROUNDS=50
set LOCAL_UPDATES=5
set ACTIVE_INDICATORS=1,1,1,1,1
set RING_ALPHA=0.2
python src/ring_federated_qcmdp_training.py --data-path C:\path\to\your\data.csv
```

Example on PowerShell:

```powershell
$env:FL_ROUNDS=50
$env:LOCAL_UPDATES=5
$env:ACTIVE_INDICATORS="1,1,1,1,1"
$env:RING_ALPHA=0.2
python .\src\ring_federated_qcmdp_training.py --data-path C:\path\to\your\data.csv
```

## Output

Typical outputs include:

- model checkpoints,
- training convergence CSV files,
- reward convergence plots,
- constraint-cost convergence plots,
- operation history CSV files,
- voltage heatmaps,
- line loading plots,
- network metric CSV files.

## Notes

- This repository is organized to match the paper formulation instead of preserving older experimental naming.
- The public environment implementation is simplified for research transparency and algorithm inspection.
- Exact figure-level reproduction may still require aligned data, configuration, random seed control, and evaluation protocol outside this repository.

## License

This project is released under the MIT License. See `LICENSE` for details.

## Citation

If you use this code, please cite the corresponding paper.


