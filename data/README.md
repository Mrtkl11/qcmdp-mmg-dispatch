# Dataset

The experiments use a local `Environment_data_2018.csv` containing 15-minute renewable generation, residential demand, and electricity-price series. The columns are `time`, `household_power`, `solar_power`, `wind_power`, and `EUR/kWh`.

The paper describes the source families as Renewables.ninja data derived from NASA MERRA-2, ENTSO-E day-ahead prices, and BLEMdataGlimpse residential demand. The full merged dataset is excluded from Git so that users can obtain the source data under the applicable upstream terms. A small synthetic schema fixture is provided under `tests/fixtures/` for automated tests only.
