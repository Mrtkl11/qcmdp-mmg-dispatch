$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $PSScriptRoot
$DataPath = if ($args.Count -ge 1) { $args[0] } else { Join-Path $RootDir "data\Environment_data_2018.csv" }
$QcmdpConfig = if ($args.Count -ge 2) { $args[1] } else { Join-Path $RootDir "configs\experiment_config.template.json" }
$BaselineConfig = if ($args.Count -ge 3) { $args[2] } else { Join-Path $RootDir "configs\baselines.template.json" }

Push-Location $RootDir
try {
    python -m src.qcmdp_single_mg_training --data-path $DataPath --config-json $QcmdpConfig --mg-id 1 --output-dir (Join-Path $RootDir "results\single_mg\qcmdp")
    foreach ($Algorithm in @("a2c", "ppo", "qppo", "sac", "constrained_ppo")) {
        python -m baselines.runner --algorithm $Algorithm --data-path $DataPath --config-json $BaselineConfig --mg-id 1 --output-dir (Join-Path $RootDir "results\single_mg\$Algorithm")
    }
    python -m src.ring_federated_qcmdp_training --data-path $DataPath --config-json $QcmdpConfig --output-dir (Join-Path $RootDir "results\federated\qcmdp")
    foreach ($Algorithm in @("a2c", "ppo", "qppo", "sac", "constrained_ppo")) {
        python -m baselines.federated --algorithm $Algorithm --data-path $DataPath --config-json $BaselineConfig --output-dir (Join-Path $RootDir "results\federated\$Algorithm")
    }
}
finally {
    Pop-Location
}
