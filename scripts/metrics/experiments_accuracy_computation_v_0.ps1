# $expname = "training_20241028_195657"  # the best for far
# $TileId = "S2B_31TEN_20230301_0_L2A"
# $TileId = "S2B_31TDJ_20240921_0_L2A"
# $TileId = "S2A_31TDK_20240916_0_L2A"
# "S2B_33UUU_20230421_0_L2A"
# "S2A_32UQB_20240430_0_L2A"
# "S2B_32UPE_20240309_0_L2A" "S2B_32UPD_20240306_0_L2A"
# "S2B_32UPE_20240309_0_L2A" "S2A_32UQB_20240430_0_L2A"
# "S2B_10SFJ_20230424_0_L2A"
# "S2A_33UVR_20240427_0_L2A"
# "S2A_33UUU_20240427_0_L2A"
# "S2A_32UPB_20240430_0_L2A" # Thuringia, Germany
# "S2B_31UFT_20230301_0_L2A"
# "S2A_30TYR_20220430_0_L2A"
# "S2A_33UYT_20220412_0_L2A"
# "S2A_30TYR_20220430_0_L2A" # "S2A_31TDJ_20220417_0_L2A"
# "S2A_31TEN_20220629_0_L2A"
# "S2A_31TEN_20220629_0_L2A" # "S2A_31TEN_20220530_0_L2A"
# "S2A_31TDJ_20220417_0_L2A"
# "S2A_31UDQ_20220324_0_L2A" # "S2A_31TFL_20220417_0_L2A"
# "S2A_31TEM_20220417_0_L2A"
# "S2A_31TEN_20220417_0_L2A" # "S2B_30UWU_20220421_0_L2A"
# "S2B_30UXV_20230403_0_L2A"
# "S2A_30UWU_20230302_0_L2A" # "S2B_31TEN_20230420_0_L2A"


$expname = "workflow"
$version = "v1.0.1" # "training_20241028_195657"

$base_abs = "."
$project_root = "$base_abs\niva_check\data\accuracy\$expname"
$code_path = "$base_abs\NIVA-model\src\other"
$path_to_borders = "$base_abs\NIVA-model\data\\cadastre_metadata_v1.0.0.gpkg"

# S2A_32UPB_20240430_0_L2A (no in demo)

# S2A_10SGG_20230426_0_L2A
# S2A_32UQB_20240430_0_L2A
# S2A_33UUU_20240427_0_L2A
# S2B_32UPE_20240309_0_L2A

# https://gdal.org/en/stable/programs/gdal_translate.html
# -scale 0 3000 0 255 -exponent 1
# gdal_translate -of VRT -scale 0 3000 0 255 -exponent 1 "NETCDF:"""S2A_10SGG_20230426_0_L2A\\tile\\S2A_10SGG_20230426_0_L2A.nc""":B2" S2A_10SGG_20230426_0_L2A/tile/b2_scaled_3000.vrt

$remove_outliers = 0
$combine_regions = $true
$SimTolerance = 5


New-Item -ItemType Directory -Path (Split-Path -Parent $project_root) -ErrorAction Ignore | Out-Null

$("S2B_32UPE_20240309_0_L2A") | ForEach-Object {
    $TileId = ${_}
    $out_file = "$project_root/intersection_" + "$TileId.json"

    # download tile meta data and predicted field boundaries for the tile
    $pred_file_path_ = "$project_root\\$TileId\\predicted\\field-boundaries-$TileId-$version.geojson"
    $path_tile_meta = "$project_root\\$TileId\\meta\\$TileId.json"
    # path to convert GeoJson to geopackage for more convenient spatial op
    $pred_file_path = "$project_root\\$TileId\\predicted\\field-boundaries-$TileId-$version.gpkg"

    if (!(Test-Path $pred_file_path)) {
    # save to gpkg format
    # python -c "import geopandas as gpd; data = gpd.read_file($pred_file_path_); data.to_file($pred_file_path)"
    python "$code_path/convert2gpkg.py" -i $pred_file_path_ -o $pred_file_path
    }
    # TODO automatically download predicted field boundaries + tile metadata
    New-Item -ItemType Directory -Path (Split-Path -Parent $path_tile_meta) -ErrorAction Ignore | Out-Null
    New-Item -ItemType Directory -Path (Split-Path -Parent $pred_file_path) -ErrorAction Ignore | Out-Null

    if (!(Test-Path $out_file)) {
    Write-Output "Find intersection between tile and cadastre data $out_file"

    python "$code_path\find_cadastre_intersect.py" --cadastre_meta $path_to_borders --tile_meta $path_tile_meta --output $out_file
    }
    Get-Content -Raw $out_file | ConvertFrom-Json | ForEach-Object {
        # "region", "url", "source_path"
        $record = ${_}
        Write-Output "$record"

        $SubName = ($record.url  -split '/')[-1]
        $filename = ($SubName  -split '\.')[0]
        $Grid = $TileId[4..8] -join ''
        $GridName = "_$Grid" + "_"
        $cadastre_tile_path = "$project_root/$TileId/tile/cadastre_" + "$remove_outliers/" + "$filename$GridName$SimTolerance" + ".gpkg"

        # preprocess the cadastre
        ./cadastre_proc.ps1 $TileId $expname $record.source_path $record.url $pred_file_path $project_root $code_path $remove_outliers $SimTolerance $cadastre_tile_path
        #./accuracy_comp.ps1 $TileId $expname $record.source_path $record.url $pred_file_path $project_root $code_path
        if (!($combine_regions)) {
            # compute accuracy metrics for the regional cadastre data
            $final_file_name = "region_" + $filename + "_expname_" + $expname + "_r_" + "$remove_outliers-$version"
            $metrics_path = "$project_root/$TileId/metrics_$final_file_name.csv"
            ./accuracy_comp_only.ps1 $TileId $pred_file_path $project_root $code_path $cadastre_tile_path $metrics_path $path_tile_meta
        }
    }

    if ($combine_regions) {
        $cadastre_folder = "$project_root/$TileId/tile/cadastre_" + "$remove_outliers"
        $cadastre_tile_path = "$project_root/$TileId/tile/cadastre_combined_$remove_outliers.gpkg"

        if (!(Test-Path $cadastre_tile_path)) {
            Write-Output "Combine cadastre data for different regions $cadastre_tile_path"
            python "$code_path/combine_cadastre.py" --input_folder $cadastre_folder --output $cadastre_tile_path
        }
        # compute accuracy metrics for the combined cadastre of regions data
        $filename = "combined"
        $final_file_name = "region_" + $filename + "_expname_" + $expname + "_r_" + "$remove_outliers-$version"
        $metrics_path = "$project_root/$TileId/metrics_$final_file_name.csv"
        Write-Output "Computing accuracy scores $metrics_path"
        ./accuracy_comp_only.ps1 $TileId $pred_file_path $project_root $code_path $cadastre_tile_path $metrics_path $path_tile_meta
    }
}