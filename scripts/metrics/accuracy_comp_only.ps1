
Param ([string] $TileId = "S2B_31UFT_20230301_0_L2A",
    [string]$pred_file_path = "\contours\S2B_31UFT_20230301_0_L2A\merged.geojson",
    [string]$project_root = "\niva_check\data\accuracy",
    [string]$code_path = "\NIVA-model\src\other",
    [string]$cadastre_tile_path = ".",
    [string]$metrics_path = "*.csv")


$eopatches_folder = "$project_root/$TileId/eopatches"


if (!(Test-Path $metrics_path)) {
Write-Output "Compute accuracy for the predicted $pred_file_path and cadastre $cadastre_tile_path"
python "$code_path\accuracy_computation_sim.py" --cadastre_tile_path $cadastre_tile_path `
    --eopatches_folder $eopatches_folder --pred_file_path $pred_file_path --metrics_path $metrics_path
}
