# Input schema and data preparation

The final model uses fourteen conditioning factors:
`DEM,slope,aspect,plan,profile,NDVI,land_use,roughness,road,river,rx5day,raincv,soil_clay,soil_silt`.
`landslide` is 0/1; `pointid` identifies a record; `cell_id` identifies a grid cell;
`fold` is the frozen outer spatial region (0–4). Records sharing a cell must share a fold.
`spatial_block_id` is required for the conditional spatial-block bootstrap.

Aspect is encoded as sine/cosine and a flat indicator. LULC is categorical.
The original functions fit transformations within the relevant training subsets.
Pass raw factor values so transformations can be fitted within training subsets.
Rx5day and daily-rainfall CV use the archived long-term definition. Clay and silt
refer to the 30–60 cm soil layer. `building`, `temyear` and `rainfall` are auxiliary
historical/ablation columns, not extra main-model factors.

The source release does not redistribute the original inventory, provider rasters,
author-local processed sample table, fitted estimators or full-domain scores.
Provide the lawful processed modelling table to rerun training. No synthetic or study-input dataset is bundled.

The optional raw-data path uses `research/config/paths.json`. Raster sampling,
background buffers, numeric precision and frozen cell/fold assignments must agree
with the supplied analysis settings when reproducing the case study. The source
contains the preparation and coverage functions, while ArcGIS or rasterio workflows
require the corresponding user-supplied geospatial inputs.
