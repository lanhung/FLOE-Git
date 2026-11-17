# FLOE v1.0: Sentinel-1 EW scenes with ice-chart floe-size labels around Greenland, 2022–2026

3224 Sentinel-1 Extra Wide swath scenes (1 January 2022 to 9 November 2026) paired with the regional
ice charts of the Greenland Ice Service of the Danish Meteorological Institute, with floe-size,
concentration and stage-of-development labels in the image geometry and the full egg code of every
chart polygon; 289 GB in 3224 files. Described in: [dataset paper, reference to be added on publication].

## Licence and attribution

Released under the Creative Commons Attribution 4.0 International licence (CC BY 4.0, `LICENSE.txt`).
When you use the data, please cite the dataset paper and credit:

- ice charts: **The Greenland Ice Service, Danish Meteorological Institute** (published here with
  their written permission, November 2026);
- satellite data: **Contains modified Copernicus Sentinel data 2022–2026**.

## Contents

| Path | Content |
|---|---|
| `data/<year>/<product>.nc` | one netCDF4 file per scene, named after its Sentinel-1 product |
| `scenes.csv` | one row per scene: product, file, year, start time, mission, charts used, largest chart–frame time gap (min), strict-subset flag (all charts within 5 min of the frame), benchmark role, size, MD5 |
| `splits/` | the benchmark split: `fold{0-4}_train.json`, `fold{0-4}_val.json`, `test.json` (2026), `excluded.json` (with reasons), `split_report.md` |
| `MD5SUMS` | checksums of all scene files (`md5sum -c MD5SUMS`) |

## A scene file

Rasters in the original SAR geometry at 80 m (dimensions `y`, `x`; typically about 5000 × 5200):

| Variable | Meaning |
|---|---|
| `nersc_sar_primary`, `nersc_sar_secondary` | HH and HV backscatter, noise-corrected, standardised (z-score; mean and standard deviation in the variable attributes) |
| `sar_incidenceangle` | incidence angle, standardised |
| `global_valid_mask` | valid SAR pixel (1) |
| `FLOE`, `SIC`, `SOD` | floe-size, concentration and stage-of-development classes by the AutoICE rules (fill value in `chart_fill_value`) |
| `polygon_id` | chart polygon covering the pixel, an index into the `poly_*` table |
| `chart_edge_distance` | distance to the nearest polygon edge |

Per chart polygon (dimension `polygon`): `poly_codes` (SIGRID-3 egg code, field names in the global
attribute `poly_codes_fields`), `poly_composition` (floe-size composition, dimension `floe_class`:
water, cake, small, medium, big, vast), `poly_floe`, `poly_sic`, `poly_sod`, `poly_unknown`,
`poly_chart`. Georeferencing tie points (dimensions `grid_row`, `grid_col`): `sar_grid_lat`,
`sar_grid_lon`, `sar_grid_inc`, `sar_grid_height`. The global attributes hold the product name,
times, pass, mission, processor version, the charts used, the time gaps and the quality numbers.

Floe-size classes: 0 water, 1 cake ice, 2 small, 3 medium, 4 big, 5 vast floes, 6 icebergs (ignore
when training floe size), 255 no class (land, outside the charts, or a polygon in which no floe
class reaches 65 % of the ice).

Read with any netCDF4 reader, e.g. `xarray.open_dataset(path)`; no plugins are needed.

## Benchmark protocol

Five grouped folds over the strict subset of 2022–2025, the year 2026 held out as the test set, and
the scores listed in the dataset paper. Report all seeds and all folds; do not choose models on 2026.

## Rebuilding

The code that builds, checks, splits and evaluates the dataset, and rebuilds it from the Sentinel-1
and DMI archives, is at https://github.com/lanhung/FLOE-Git [DOI of the code release to be added].
The original chart shapefiles are not included; they are available from
https://download.dmi.dk/public/ICESERVICE/SIGRID3/ (chart names in `scenes.csv`).
