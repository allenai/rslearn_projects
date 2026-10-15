# PASTIS2 — France-wide PASTIS-style crop SITS dataset

Extends the [PASTIS](https://github.com/VSainteuf/pastis-benchmark) benchmark (4 Sentinel-2
tiles over metropolitan France) to **all French parcels** — metropole + Corsica + the **5
overseas départements (DROM)**: Guadeloupe, Martinique, Guyane, Réunion, Mayotte. Labels
come from the **RPG** (Registre Parcellaire Graphique); imagery is materialized with rslearn
from the **Microsoft Planetary Computer**.

Semantic labels, **26 classes** (PASTIS's 18 + 7 tropical DROM classes; 0 = background).

## Imagery: faithful to PASTIS (structure), from Planetary Computer (source)
We match PASTIS's acquisition design and relax only the provider:

| Aspect | PASTIS | PASTIS2 (`config_dense.json`) |
|---|---|---|
| Temporal | 38–61 **individual** acquisitions (irregular) | **dense** — all scenes over the span (`space_mode=INTERSECTS`, `max_matches=150`); ~74/window typical |
| Date range | Sep 2018 – Nov 2019 | Sep 2018 – Nov 2019 (`growing_season()` in `build_windows.py`) |
| Cloud filter | drop tiles > 90 % cloud, keep partial cloud | `eo:cloud_cover ≤ 90` (matches PASTIS) |
| Bands / res | 10 L2A bands @ 10 m | same 10 (B02–B12, B8A) @ 10 m |
| Patch | 128 × 128 | 128 × 128 (1.28 km cells) |
| Source / correction | THEIA (MAJA) | **Planetary Computer (Sen2Cor)** — the one intentional difference |
| + Sentinel-1 | — | dense VV/VH added (`sentinel1` layer), an extension beyond PASTIS |

> THEIA/MAJA has no rslearn data source (geodes STAC requires auth + isn't per-window
> streamable), so PC/Sen2Cor is the closest available; reflectance differs from MAJA.

## Setup
Use the rslearn venv (has `rslearn[extra]` — planetary_computer, pystac_client, pyogrio),
e.g. `/weka/dfive-default/piperw/dev/olmoearth_pretrain/.venv/bin/python`. RPG download also
needs `pip install py7zr` (IGN `.7z` archives). The `data/` artifacts are **not** in git
(metropole alone is > 5 GB); regenerate them via the pipeline below.

## RPG source data (IGN geoplateforme, RPG 2019)

| Territory | Parcels | Positive-class |
|---|---|---|
| Metropole (national) | 9,604,463 | 7,934,143 |
| Guadeloupe | 24,613 | 20,354 |
| Martinique | 11,485 | 9,063 |
| Guyane | 3,289 | 1,953 |
| Réunion | 18,207 | 16,194 |
| Mayotte | 4,024 | 530 |
| **TOTAL** | **9,666,081** | **7,982,237 (83 %)** |

## Pipeline (files in this branch)

| Step | Script / file | What it does |
|---|---|---|
| 0 | `build_pastis_rpg_map.py` → `pastis_rpg_class_map.json` | RPG `CODE_CULTU` → class id 0–25 (uses `data/rpg_culture_codes.csv`) |
| 1 | `download_rpg.py --year 2019` | download each territory's RPG → `data/rpg/<key>.gpkg` (geometry + `code_cultu` + `class_id`); URLs in `territories.py` |
| 2 | `build_windows.py --dataset <ds> --year 2019` | stratified 128×128 @10 m windows (per-window UTM), group `rpg_<year>`, time range Sep 2018–Nov 2019 |
| 3 | `config_dense.json` | the rslearn dataset config: `sentinel2` (dense, 10-band) + `sentinel1` (dense VV/VH) + `label` |
| 4 | `launch_pastis2_dense.sh` **or** `rslearn dataset prepare/materialize` | fetch + materialize the dense S2/S1 time series |
| 5 | `rasterize_labels.py --dataset <ds> --group rpg_2019` | burn parcels → per-window `label` raster (class_id), aligned to the S2 grid |

`config_dense.json` is the dataset config — copy it to your dataset root as `config.json`
(`build_windows`/`rslearn` read `<ds>/config.json`).

## Build a dataset

```bash
PY=/weka/dfive-default/piperw/dev/olmoearth_pretrain/.venv/bin/python
DS=data/national_ds

# 0-1. class map + RPG (once).  Restrict to a subset by pointing PASTIS2_RPG_DIR at a dir
#      with only the desired <key>.gpkg (metropole absent -> skipped).
$PY build_pastis_rpg_map.py            # -> pastis_rpg_class_map.json
$PY download_rpg.py --year 2019        # -> data/rpg/*.gpkg

# 2. windows.  --target-total = metropole sample size; --per-territory-min = per DROM.
#    (Set --target-total 0 + a large --per-territory-min to take ALL DROM cells, no metropole.)
mkdir -p $DS && cp config_dense.json $DS/config.json
$PY build_windows.py --dataset $DS --year 2019 --target-total 20000 --per-territory-min 1000

# 3-4. materialize (dense S2 + S1).  Local:
$PY -m rslearn.main dataset prepare     --root $DS --workers 16 --retry-max-attempts 8 --retry-backoff-seconds 30
$PY -m rslearn.main dataset materialize --root $DS --workers 16 --retry-max-attempts 8 --retry-backoff-seconds 30
#    ...or fan out over N Beaker jobs (gantry) for scale:
#    NUM_JOBS=8 DS=$PWD/$DS bash launch_pastis2_dense.sh

# 5. labels
$PY rasterize_labels.py --dataset $DS --group rpg_2019
```

### Materialize at scale — `launch_pastis2_dense.sh`
Fans the prepare+materialize out over N Beaker jobs via **gantry**. Every job runs the same
`rslearn dataset <verb> --root <ds> --group rpg_2019`; rslearn shuffles window order and
skips windows already marked completed, so N jobs distribute the work lock-free (resumable —
re-run to fill gaps). Dense is heavy (~74 S2 + dense S1 per window ≈ 40 MB/window), so start
with `NUM_JOBS=8–12` due to PC rate limits.

## Notes
- **`num_classes = 26`** (PASTIS-18 + 19 Sugarcane, 20 Banana, 21 Pineapple, 22 Vanilla,
  23 Tropical tuber, 24 Ylang-ylang, 25 Coffee/Cacao). The DROM extension gives Mayotte
  real labels (ylang/banana/vanilla…) instead of ~0 %. `ACA` "autre culture non précisée"
  stays Background. Any model/eval must use `num_classes = 26`, not 19.
- **`pastis_rpg_class_map.json` → `flagged_for_review`** lists judgement calls to verify
  against PASTIS's official nomenclature.
- **Spatial splits**: PASTIS used 5 scene-disjoint folds; assign a proper spatial/geographic
  holdout at training time (each temperate class is metropole-only and each tropical class is
  DROM-only, so a naive split can strand a whole class on one side).
- **Tensor conversion + viz**: the `pastis_r`-style 12-month 64² eval-tensor exporter
  (`make_tensors.py`) and the coverage-map / QA-viz tooling live on the fuller
  `piperw/pastis2` branch; this branch keeps only the window-building + materialize + label
  pipeline.
