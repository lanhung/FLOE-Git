# AI4Arctic / AutoICE benchmark (public, CC BY 4.0)

Official manifests (name, size, MD5, Figshare file id) fetched from the Figshare API
on 2026-11-10 for:

- `manifest_rtt_train_v3.csv`: Ready-To-Train AI4Arctic Sea Ice Challenge Dataset, v3
  (doi 10.11583/DTU.21316608), 512 scenes + manual, 60.2 GB.
- `manifest_rtt_test_v2.csv`: Ready-To-Train Test Dataset, v2 (with reference charts),
  20 scenes, 2.5 GB.
- `manifest_raw_v3.csv`: Raw dataset, v3 (40 m, chart polygon codes), 22 zips, 261 GB.

Figshare/DTU downloads are behind a bot challenge and blocked from the GPU server, so
the ready-to-train data were taken from the TorchGeo re-host
`torchgeo/ai4artic-sea-ice-challenge` (Hugging Face) and every extracted file is
checked against these official MD5s (`tools/data/fetch_ai4arctic.sh`).

Cite: Buus-Hinkler et al. (2022), AI4Arctic Sea Ice Challenge Dataset,
doi 10.11583/DTU.c.6244065; Stokholm et al. (2024), The Cryosphere 18, 3471.


## Conversion to this project's format (`tools/data/convert_ai4arctic.py`)

- AutoICE standardisation (toolkit `misc/global_meanstd.npy`, read by static
  pickle-opcode parsing, never unpickled): HH −14.5083 / 5.6597 dB, HV −24.7012 /
  4.7468 dB, incidence angle 33.9980 / 8.3223°. This archive: HH −13.9018 / 8.3349,
  HV −27.0105 / 9.4454, incidence angle 34.2622 / 7.7648. Inputs are converted to
  physical units, HH/HV clipped to [−35, 5] dB as in this archive, and re-standardised
  with this archive's statistics, so both datasets share one radiometric space.
- Labels: AutoICE FLOE classes 0–5 use the same SIGRID-3 mapping as this archive;
  class 6 (icebergs, growlers) → 255 (ignored here); invalid SAR → 255.
  Remaining definition difference: AutoICE dominance threshold 0.65, this archive's
  base config 0.5.
- No GLCM bands in AutoICE: models use HH, HV, incidence angle, valid mask + pol
  channel (`configs/AI4Arctic/`, `configs/FLOE/res80_noglcm.py`).
- Known processing differences that remain (report as domain shift): AutoICE SAR is
  not Refined-Lee filtered; 40 → 80 m by averaging; 2018–2021 vs 2022–2026.
- Splits: `test.json` = the 20 official test scenes (reference charts); `val.json` =
  ~10% of training acquisition days (stable hash of the date), for model selection
  only; `train.json` = the rest. Generated on the server by
  `tools/data/prepare_ai4arctic.sh`.
