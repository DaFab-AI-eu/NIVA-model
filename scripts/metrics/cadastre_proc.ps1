
Param ([string] $TileId = "S2B_31UFT_20230301_0_L2A",
    [string]$expname = "training_20241028_195657",
    [string]$source_path = "nl_ref.parquet",
    [string]$url_cadastre = "https://data.source.coop/fiboa/nl-ref/nl_ref.parquet",
    [string]$pred_file_path = "niva_check\data\\temporary\contours\S2B_31UFT_20230301_0_L2A\merged_simplified.gpkg",
    [string]$project_root = "\niva_check\data\accuracy",
    [string]$code_path = "\NIVA-model\src\other",
    [int]$remove_outliers = 0,
    [int]$SimTolerance = 5,
    [string]$cadastre_tile_path = "*.gpkg")

# input paths to be defined by user

$tile_path = "$project_root/$TileId/tile/$TileId.nc"
if (!(Test-Path $tile_path)) {
Write-Output "Download tile for tile_id $TileId"
python "$code_path\tile_download_by_id.py" --tile_id $TileId --project_root $project_root
}

# download cadastre data for accuracy estimations
$SubName = ($url_cadastre  -split '/')[-1]
$filename = ($SubName  -split '\.')[0]
$out_file = "$project_root/$SubName"

if (!(Test-Path $out_file)) {
Write-Output "Download cadastre data from $url_cadastre"
try {
        Invoke-RestMethod -Uri $url_cadastre `
            -OutFile $out_file
    }
    catch {
        Write-Error "$url_cadastre not found."
        Exit
    }
}

if (!(Test-Path "$project_root/$source_path")) {
Write-Output "Extract .7z archive $out_file into $project_root"
# Expand-Archive -LiteralPath $out_file -DestinationPath $project_root
python "$code_path\z_extract.py" --input $out_file --output $project_root
}

$cadastre_path = "$project_root\$source_path"
$tile_meta = "$project_root\$TileId\tile\metadata.gpkg"

New-Item -ItemType Directory -Path (Split-Path -Parent $cadastre_tile_path) -ErrorAction Ignore | Out-Null


if (!(Test-Path $cadastre_tile_path)) {
Write-Output "Preprocess cadastre data - saving it only for tile grid. `
Cadastre final output path for the grid $cadastre_tile_path with remove_outliers $remove_outliers"
python "$code_path\cadastre_preprocess.py" --cadastre_tile_path $cadastre_tile_path `
    --cadastre_path $cadastre_path --sim_tolerance $SimTolerance --tile_meta $tile_meta --remove_outliers $remove_outliers
}

$eopatches_folder = "$project_root/$TileId/eopatches"
# create base eopatches (folder with bbox for patches)
if (!(Test-Path $eopatches_folder)) {
Write-Output "Create basic patches for bbox definition in folder $eopatches_folder"
python "$code_path\create_patches.py" --tile_path $tile_path --eopatches_folder $eopatches_folder
}

