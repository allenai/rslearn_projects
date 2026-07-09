# PASTIS2 — France-wide PASTIS-style crop dataset

Extends PASTIS from its 4 Sentinel-2 tiles to **all French parcels** (metropole + Corsica
+ the 5 overseas départements) using the RPG as labels and rslearn to materialize
Sentinel-2 time series (via **Planetary Computer**). Semantic-only, **25 classes** (PASTIS's
18 + 6 tropical DROM classes), output in the 12-month 64² `pastis_r` eval-tensor format.

## Setup / reproduce (for others cloning this)
The `data/` artifacts are **not** in git (5+ GB; regenerable). To reproduce:
1. Use the **rslearn_projects venv** (has `rslearn[extra]` — incl. planetary_computer,
   pystac_client, pyogrio — plus `rslp`): `.../rslearn_projects/.venv/bin/python`.
   Also `pip install py7zr` (for the IGN `.7z` archives).
2. Regenerate the staged RPG (downloads ~3.6 GB metropole + small DROMs from IGN):
   `python download_rpg.py --year 2019` → writes `data/rpg/*.gpkg`.
3. Then follow **National generation** below.

## RPG data — ALL 6 territories downloaded + staged ✅
`data/rpg/*.gpkg` (normalized: geometry + `code_cultu` + `class_id`), from IGN
geoplateforme RPG 2019:

| Territory | Parcels | Positive-class |
|---|---|---|
| Metropole (national) | 9,604,463 | 7,934,143 |
| Guadeloupe | 24,613 | 20,354 |
| Martinique | 11,485 | 9,063 |
| Guyane | 3,289 | 1,953 |
| Réunion | 18,207 | 16,194 |
| Mayotte | 4,024 | 530 |
| **TOTAL** | **9,666,081** | **7,982,237 (83%)** |

Reproduce: `python download_rpg.py --year 2019` (URLs wired in `territories.py`).

## Pipeline

| Phase | Script | What it does |
|---|---|---|
| a | `build_pastis_rpg_map.py` → `pastis_rpg_class_map.json` | RPG `CODE_CULTU` → class id 0–24 (18 PASTIS + 6 tropical DROM; 0=background) |
| 1 | `download_rpg.py --year 2019` | download each territory's RPG, write `data/rpg/<key>.gpkg` (geometry + code_cultu + class_id) |
| 3 | `build_windows.py --dataset <ds> --year 2019` | stratified 128×128@10m windows over all territories (`use_utm=True`), group `rpg_<year>` |
| 5 | `rslearn dataset prepare/ingest/materialize --root <ds>` | fetch + materialize the Sentinel-2 monthly time series (layer `sentinel2`) |
| 4 | `rasterize_labels.py --dataset <ds> --group rpg_2019` | burn parcels → per-window `label` raster (class_id), aligned to the window grid |
| 6 | `make_tensors.py --dataset <ds> --group rpg_2019 --out data/tensors` | rslearn windows → 12-month, 13-band (B1/B9/B10 imputed), 64² tensors + targets, in the `pastis_r` eval layout |

`config.json` declares the three layers: `sentinel2` (raster time series, 10 PASTIS bands),
`crop_parcels` (vector, LocalFiles over `data/rpg/`), `label` (raster, written by Phase 4).

## Smoke run — VALIDATED end-to-end ✅
A 1-window smoke test (4 synthetic parcels near Toulouse; `smoke_rpg/`, `smoke_ds/`) run
with the `dev/olmoearth_pretrain/.venv` python confirmed the full mechanism:
- **Windows**: grid tiling + per-window UTM auto-selection (window CRS EPSG:32631).
- **S2 prepare**: 12 monthly item groups over Sep2018–Aug2019 (the PASTIS 12-step structure).
- **S2 materialize**: per-timestep geotiffs `layers/sentinel2[.i]/…/geotiff.tif`, each
  **10 bands, 128×128, uint16, 10 m, UTM 31N**, real reflectance — one dir per month.
- **Labels**: `rasterize_labels.py` wrote a 128×128 uint8 mask pixel-aligned to the S2 grid;
  the 4 parcels rasterized to the correct class_ids (1/2/3/4), 900 px each.
- **Phase 6**: `make_tensors.py` produced `data/tensors/pastis2_test/` — `s2_images/{i}.pt`
  (12,13,64,64) float32, `targets.pt` (N,64,64) in {0..18}, `months.pt` (N,12) on the
  201809..201908 grid — i.e. the exact `pastis_r` eval layout. Missing months zero-filled.

Run it yourself:
```
PY=/weka/dfive-default/piperw/dev/olmoearth_pretrain/.venv/bin/python
PASTIS2_RPG_DIR=$PWD/smoke_rpg $PY build_windows.py --dataset smoke_ds --year 2019 --target-total 2 --per-territory-min 0
$PY -m rslearn.main dataset prepare    --root smoke_ds --workers 0
$PY -m rslearn.main dataset materialize --root smoke_ds --workers 0
PASTIS2_RPG_DIR=$PWD/smoke_rpg $PY rasterize_labels.py --dataset smoke_ds --group rpg_2019
```

## Real-data run — VALIDATED on Corsica (R94-2019) ✅
Ran the full pipeline on **real RPG** (not synthetic): downloaded from the IGN
geoplateforme (`data.geopf.fr/telechargement`, discovered via its capabilities API),
Corsica 2019 (18 MB, 43,621 parcels, 38,554 with a positive PASTIS class).
`download_rpg.py` → `build_windows.py` → prepare → materialize → `rasterize_labels.py`
→ `make_tensors.py` → `viz_tensors.py` all ran end-to-end. Real parcels rasterize to
real, varied classes (meadow, grapevine, orchard, winter/spring barley, triticale,
leg-fodder, soft wheat) with realistic cadastral field geometry, pixel-aligned to real
Sentinel-2 — confirmed visually (`corsica_patch3.png`).

### S2 source: use Planetary Computer (RESOLVED the coverage bug ✅)
Switching the `sentinel2` layer from `aws_sentinel2_element84` to
`rslearn.data_sources.planetary_computer.Sentinel2` (harmonize + `eo:cloud_cover<70` +
`sort_by`, 10 PASTIS bands) **fixed the coverage problem**: a plain 12-month window now
materializes **12/12 monthly composites for every window** (verified on 2 Corsica windows,
incl. the one that got 0 with Element84). `make_tensors` then reports filled slots
min=max=mean=12.0 — a full, gap-free 12-month series (confirmed visually in `pc_patch.png`).
Requires `pip install planetary_computer pystac_client`. The over-fetch machinery
(`--fetch-months 18/24` + nearest-fill) is now just an optional safety net, not required.

### (Historical) Element84 direct-path bug that motivated the switch:
- **Materialize writes only a prefix of the monthly composites.** In a 24-month
  over-fetch run, prepare found 23 candidate composites spanning 201710..201908, but
  materialize wrote only `group_idx` 0–6 (the 7 *most-recent* months, Mar–Aug 2019) and
  silently dropped the other 16. Same shape at 12 months (wrote ~5). So coverage is
  capped at ~5–7 clustered recent months regardless of the request window.
- **Over-fetch (`--fetch-months 18/24`) + nearest-fill is implemented and mechanically
  fills all 12 slots, but can't fix this** — with only ~7 clustered composites it just
  duplicates them across slots (e.g. 201903 smeared over 7 winter slots). Useless until
  materialize stops dropping groups.
- Some windows materialize **0** composites entirely (window 51584 in both runs).
- `ingest:true` is NOT a fix (it hangs on this direct source).
- **Next (durable fix):** switch the S2 layer to `planetary_computer.Sentinel2` (the
  source rslearn's own multi-period example uses) or `gcp_public_data.Sentinel2`, which
  likely don't have the direct-path prefix-drop; re-check that all 12 months materialize.

DROM note: the taxonomy was **extended beyond PASTIS's 18** with 6 common French-DROM
tropical classes — 19 Sugarcane, 20 Banana, 21 Pineapple, 22 Vanilla, 23 Tropical tuber,
24 Ylang-ylang (**num_classes = 25**). With this, Mayotte (D976-2019, 4,024 parcels) now
gets 13% positive labels (banana 159, vanilla 131, ylang 119, tuber 47, pineapple 27,
orchard 26, sugarcane 5…) instead of 0%; the remainder is `ACA` "autre culture non
précisée" which correctly stays Background. Any model/eval on pastis2 must use
num_classes = 25 (not 19).

## National generation (end-to-end)
```
DS=/weka/.../pastis2/national_ds          # shared weka root; config.json = PC sentinel2 + label
# 1. windows over all 6 territories (stratified; heavy metropole read, run once)
python build_windows.py --dataset $DS --year 2019 --target-total 20000 --per-territory-min 200
# 2+3. prepare + materialize S2, fanned out over Beaker (rslearn_projects venv):
RSLP=/weka/dfive-default/piperw/rslearn_projects/.venv/bin/python
$RSLP launch_national.py --dataset $DS --group rpg_2019 --image <rslearn+pc-image> --num-jobs 16 --step both
# 4+5. after materialize: labels + eval tensors (cheap)
python rasterize_labels.py --dataset $DS --group rpg_2019
python make_tensors.py --dataset $DS --group rpg_2019 --out data/tensors_national --year 2019
```

## Remaining TODO (before national generation)
- **Stratified sampler — DONE ✅** — `build_windows.sample_cells()` bins positive parcels
  into 1.28 km cells (one window each) and draws cells with probability ∝ rarity of their
  rarest class (inverse global class frequency, `RARITY_POWER`), so minority crops aren't
  drowned out by meadow/cereal. Windows are created center-based (`window_size`), one per
  cell, per-territory UTM. Validated: on Réunion a 150-cell stratified draw covers all 13
  classes vs 10 for a random draw; DROM windows land in the correct UTM zones. Per-territory
  volume is set by `--target-total` (metropole) and `--per-territory-min` (each DROM).
  Optional future refinement: add an admin (département) join for explicit geographic strata.
- **Beaker parallel materialize — DONE ✅** — `launch_national.py` wraps the repo helper
  `rslp.common.beaker_data_materialization.launch_jobs` to fan prepare + materialize out
  over N Beaker jobs. All jobs run the *same* `rslearn dataset <verb> --group rpg_2019`
  against one weka dataset root; rslearn shuffles window order per job and skips windows
  already marked `completed` (per-window marker files), so the jobs distribute the work
  lock-free and disjointly. Run with the rslearn_projects venv. Default image
  `favyen/rslpomp20260702a` is **verified** to import planetary_computer + pystac_client +
  rslearn's PC data source (any rslp image built from `rslearn_projects/Dockerfile` has
  `rslearn[extra]`, which includes both — no install step needed). rasterize_labels +
  make_tensors run after (cheap, no download). Underlying helper is also exposed as
  `python -m rslp.main common launch_data_materialization_jobs`. NOT yet launched — volume
  (`--num-jobs`) is the caller's choice; start small (8–16) due to PC rate limits.
- **Phase 6 splits** — `make_tensors.split_for()` is a name-hash placeholder; use a proper
  spatial/geographic holdout (PASTIS used 5 folds) so train/test aren't adjacent.
- **Class map** — `pastis_rpg_class_map.json` `flagged_for_review` lists judgement calls to
  verify against PASTIS's official Nomenclature.
- **Missing-month fill** — moot now PC gives 12/12, but `make_tensors` nearest-fill can still
  duplicate a month if a season month is genuinely absent; revisit if it recurs.
