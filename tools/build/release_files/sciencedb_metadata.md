# Metadata to paste into the ScienceDB submission form

**Title:** FLOE v1.0: Sentinel-1 EW scenes with ice-chart floe-size labels around Greenland, 2022–2026

**Authors:** [as on the paper, with ORCID]

**Licence:** CC BY 4.0

**Keywords:** sea ice; floe size; Sentinel-1; synthetic aperture radar; ice charts; Greenland;
machine learning benchmark; semantic segmentation

**Subject:** Earth sciences / Cryosphere / Remote sensing

**Temporal coverage:** 2022-01-01 to 2026-11-09. **Spatial coverage:** waters around Greenland,
frames between 55.1° and 87.5° N and 97.3° W and 44.8° E (extent of the scene footprints).

**Description (abstract):**
FLOE pairs 3224 Sentinel-1 Extra Wide swath scenes from 1 January 2022 to 9 November 2026 with the
regional ice charts of the Greenland Ice Service of the Danish Meteorological Institute. Each scene
holds noise-corrected HH and HV backscatter and the incidence angle at 80 m in the original image
geometry, rasters of floe size, concentration and stage of development made by the rules of the
AutoICE challenge, the index of the chart polygon under every pixel, and a table with the full egg
code and the floe-size composition of every chart polygon. A scene list gives quality numbers for
every scene (chart–frame time gap, held-out georeferencing error, replaced pixels) and the benchmark
split: five folds that keep related scenes together over 2022–2025, and 2026 held out as the test
year. The data are described, and their label quality measured, in [dataset paper]. Ice charts:
The Greenland Ice Service, Danish Meteorological Institute (published with written permission).
Contains modified Copernicus Sentinel data 2022–2026.

**Related resources:** code https://github.com/lanhung/FLOE-Git [code DOI]; dataset paper [DOI
when available]; source archives https://download.dmi.dk/public/ICESERVICE/SIGRID3/ and
https://search.asf.alaska.edu

**Size and format:** 289 GB (10^9 bytes), 3224 netCDF4 files in `data/<year>/`, plus `scenes.csv`,
`MD5SUMS`, `splits/`, `README.md`, `LICENSE.txt`.

**Before publishing:** ask for the private review link (for the journal's reviewers) and register
the DOI only after acceptance of the paper, as ESSD recommends; check ScienceDB's current per-file
and per-dataset limits (scene files are 0.01–0.11 GB each, median 0.09 GB).
