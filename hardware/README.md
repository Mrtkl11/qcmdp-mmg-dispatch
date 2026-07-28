# Quantum hardware evaluation

`export_qpu_payload.py` loads trained reward and safety critics, selects minimum- and maximum-load states, and exports the RX/RY/RZ encoding angles, RY/RZ variational angles, nearest-neighbor CZ topology, local Pauli-Z expectations, and classical readout parameters.

Submit each exported circuit through an authorized QPU account and save repeated Pauli-Z measurements as a CSV with the columns `repetition`, `state_label`, `critic`, and `z0` through `zN`. `summarize_qpu_results.py` maps the hardware expectations through the trained value heads and creates the local-versus-hardware statistical comparison.

```bash
python -m hardware.export_qpu_payload --config-json configs/hardware_evaluation.template.json
python -m hardware.summarize_qpu_results --config-json configs/hardware_evaluation.template.json
```

Cloud credentials and vendor-specific submission code are intentionally kept outside version control. Do not store API keys in JSON, Python files, checkpoints, logs, or measurement tables.
