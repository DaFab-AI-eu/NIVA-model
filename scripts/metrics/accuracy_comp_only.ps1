
Param ([string] $TileId = "S2A_10SGG_20230426_0_L2A",
    [string]$pred_file_path = "field-boundaries-S2A_10SGG_20230426_0_L2A-v1.0.1.gpkg",
    [string]$project_root = "\niva_check\data\accuracy",
    [string]$code_path = "\NIVA-model\src\other",
    [string]$cadastre_tile_path = "cadastre_combined.gpkg",
    [string]$metrics_path = "*.csv",
    [string]$tile_meta_path = "*.json")


$eopatches_folder = "$project_root/$TileId/eopatches"


if (!(Test-Path $metrics_path)) {
Write-Output "Compute accuracy for the predicted $pred_file_path and cadastre $cadastre_tile_path"
python "$code_path\accuracy_computation_sim.py" --cadastre_tile_path $cadastre_tile_path `
    --eopatches_folder $eopatches_folder --pred_file_path $pred_file_path --metrics_path $metrics_path --tile_meta_path $tile_meta_path
}
