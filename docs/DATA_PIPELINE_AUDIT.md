# Audit of the original data pipeline (`preprocessing/`) and why the archive is rebuilt

`preprocessing/` (upstream commit 48d7130) produced the 417-scene archive in `Data/FLOE_datanc`.
It is a standard ESA SNAP chain, and each step is used somewhere in the literature, but several
choices make the result unfit for a floe-size study and impossible to audit or extend. Findings
below were checked in the code and, where marked, measured on the archive itself (2026-11-10).
The replacement is `tools/build` (AI4Arctic-style processing, see the end of this note).

## Geometry

1. **Equal-degree lat/lon grid (anisotropic pixels).** `process_sar.py:240` sets
   `pixelSpacingInMeter=80` on Terrain-Correction without `mapProjection`, so SNAP writes its
   default WGS84 geographic grid with 80 m expressed in degrees (0.000719 deg) on both axes.
   *Measured:* array shapes equal the footprint bounding box divided by 0.000719 deg
   (e.g. 6020 x 13170 vs 6018 x 13105 predicted; 6456 x 27125 vs 6458 x 27125). Pixels are therefore
   80 m north-south but 80 cos(lat) east-west: about 37 m at 62 N and 18 m at 76 N. Patches, the
   11 x 11 GLCM window, the variogram lags of PGSG and the 2 x 2 block mean of the resolution
   experiment all measure different ground scales in the two directions, by a factor (2-4.6) that
   varies with latitude and hence with region. Files are 2-4 times larger than an 80 m grid.
2. **No georeferencing in the NC files.** `make_labels.py:501-531` writes only the arrays; the
   x/y coordinate variables are all zeros and there is no CRS. The labels cannot be checked against
   the charts, and no other data (charts, AMSR2, compositions, SOD/SIC) can be added later.

## SAR processing

3. **HV clipped at the noise floor.** SNAP ThermalNoiseRemoval (`process_sar.py:193`, default
   clipping of negative values) followed by LinearToFromdB and the [-35, 5] dB clip in
   `normalize.py:28-30`. *Measured:* 36 % of valid HV pixels sit exactly at -35 dB in
   S1A_..._20240619T200500 (3.2 % in a 2022 scene; HH 4.8 % in the same 2024 scene). This is why
   the archive's HV statistics are -27.0 +- 9.45 dB against AutoICE's -24.7 +- 4.75 dB. ESA advises
   against clipping when the data are averaged afterwards (MPC-0392).
4. **Refined Lee 7 x 7 then bilinear terrain correction** (`process_sar.py:223-227, 240`): the
   speckle texture below ~200 m, which carries the small/medium floe information, is smoothed
   twice. No published sea-ice CNN ablation shows a benefit; AI4Arctic applies no filter.
5. Variables named `nersc_sar_*` hold ESA (SNAP) denoised data, not the NERSC product.

## Chart matching and labels

6. **Charts matched by date, within +-2 days, first chart of the day.**
   `make_labels.py:68` (`MATCH_WINDOW_DAYS = 2`) and `:163-174` pick the nearest *date* and then
   `chart_index[best_date][0]`, the first file of that day, not the chart closest in time and not
   the one covering the scene; without `FLOE_CHART_REGION` any region's chart may be used. DMI
   draws each regional chart on one Sentinel-1 image and names it with that image's time, so the
   exact chart is available; AI4Arctic matched within 5 min. East Greenland ice drifts ~15 km/day
   (up to ~70 km/day), so a chart from another pass or day misplaces labels by tens of km.
7. **Class-specific thresholds that remove big and vast floe.** `floe_label.py:17-23`: cake,
   small and medium need CT >= 6/10 and dominance >= 0.65, but big and vast need CT >= 9/10 and
   dominance >= 0.9. AutoICE uses 0.65 for every class and no CT condition. In the mixed East
   Greenland pack big/vast polygons rarely reach 9/10 and 90 %, which is consistent with the
   archive's 398 East Greenland scenes containing no big or vast pixel at all while the DMI charts
   for 2022 already contain big and vast polygons (`tools.build.audit_old` quantifies this per scene).
8. **Scene selection by class.** `prescreen_dmi_charts.py:48-49` keeps only chart dates with
   small/medium (`FLOE_PS_CLASSES=2,3`), and `make_labels.py:55` can drop whole classes
   (`KEEP_ONLY_TARGET`). Both bias class frequencies and must be reported if kept.
9. **DMI pancake read as water.** `label_definitions.py:119` maps form code 0 to class 0 (water),
   inherited from the AutoICE table; DMI writes pancake ice as 0. Any partial with a masked form
   (pancake 22, brash 1, level ice 21) also masks the whole polygon through the fast-ice test at
   `make_labels.py:475`, because all of them map to 255.
10. **Only the dominant class is kept**; polygon ids, partial concentrations and stages are lost,
    so composition-aware training and polygon-level evaluation are impossible on this archive.

## Decision

The original scripts are kept in `preprocessing/` for provenance but are not used. The archive is
rebuilt with `tools/build`:

| step | old (`preprocessing/`) | new (`tools/build`) |
|---|---|---|
| geometry | WGS84 equal-degree grid, no coordinates stored | SAR ground range, 40 m -> 80 m by 2 x 2 mean, tie-point grid stored |
| noise | SNAP, negatives clipped, [-35, 5] dB clip | ESA range x azimuth vectors, negatives kept through averaging |
| speckle filter | Refined Lee 7 x 7 | none (AI4Arctic) |
| chart match | same date +-2 days, first chart of the day | DMI chart of the same image (+-30 min) overlapping the footprint |
| labels | dominant FLOE, class-specific thresholds | SIC/SOD/FLOE with the AutoICE 65 % rule for all classes, polygon ids, egg codes, compositions |
| texture | GLCM 11 x 11 on filtered HH | none stored (computable later if an ablation needs it) |
| normalisation | dataset-fitted z-scores | AutoICE global statistics, so AutoICE scenes share the space |

The old archive stays frozen as v0 for reference runs and is not pooled with rebuilt data.
