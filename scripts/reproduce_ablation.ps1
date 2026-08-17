$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $PSScriptRoot
$DataPath = if ($args.Count -ge 1) { $args[0] } else { Join-Path $RootDir "data\Environment_data_2018.csv" }
$QcmdpConfig = if ($args.Count -ge 2) { $args[1] } else { Join-Path $RootDir "configs\experiment_config.template.json" }

Push-Location $RootDir
try {
    python -m src.ring_federated_qcmdp_training --data-path $DataPath --config-json $QcmdpConfig --federated-config-json (Join-Path $RootDir "configs\fault_tolerance.paper.json") --output-dir (Join-Path $RootDir "results\ablation\topology_reconfiguration")
    python -m src.ring_federated_qcmdp_training --data-path $DataPath --config-json $QcmdpConfig --federated-config-json (Join-Path $RootDir "configs\fault_tolerance_ablation.paper.json") --output-dir (Join-Path $RootDir "results\ablation\no_topology_reconfiguration")
}
finally {
    Pop-Location
}
