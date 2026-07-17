# MapBiomas 3k — OlmoEarth Pretrain Eval Task

Subsetting the [MapBiomas Brasil Collection 10](https://mapbiomas.org/) land-cover dataset into a compact eval task for `olmoearth_pretrain`. Two label flavors: **sparse** (expert points) and **dense** (coverage rasters).

## Dataset Source

- **Expert validation points** (~85k): `datasets/mapbiomas/metadata/mapbiomas_85k_points_validation.shp` — per-pixel expert labels with quality scores, edge flags, and map-tile IDs.
- **LULC coverage rasters**: downloaded from `https://storage.googleapis.com/mapbiomas-public/initiatives/brasil/collection_10/lulc/coverage/brazil_coverage_{year}.tif` (years 2016–2024) via `datasets/mapbiomas/download_coverage.py`.
- **Legend / hierarchy**: `datasets/mapbiomas/metadata/Codigos-da-legenda-colecao-10.csv` and `hierarchy.csv`.

MapBiomas Collection 10 provides annual land-use / land-cover maps for Brazil at ~30m resolution, derived from Landsat mosaics and validated by expert interpreters across ~85k points.

## Task Flavors

| | **Sparse** (`mapbiomas_3k_sparse`) | **Dense** (`mapbiomas_3k_dense`) |
|---|---|---|
| Label source | Expert validation point class | MapBiomas coverage raster patch |
| Label geometry | 3×3 block at window center; rest = nodata | Full 48×48 window at 10m |
| Use case | Point-level classification accuracy | Pixel-level segmentation quality |

## Subsetting Pipeline

### Step 1 — Balanced 4k point subsample (`subsampling/sample_expert_points.py`)

From the 85k expert points, select 4k (3k train / 1k val) with:
- **Year filter**: 2016–2022, best-quality only (`COUNT == 3`)
- **Class filter**: drop classes with <100 points (13, 23, 27, 30, 31, 32, 50)
- **Unique pixels**: each `TARGETID` used at most once across years
- **Water-filling quotas**: balanced class representation capped by availability
- **Stratified pick**: rarest-class-first, spread across map tiles (`CARTA_2`), 25% edge / 75% interior

### Step 2 — Dense window optimization (`subsampling/sample_dense_raster.py`)

For each sampled point, search a 512×512 neighborhood in the coverage raster, score 64 random 16×16 sub-windows by weighted minority-class pixel count, and keep the best. This shifts window centers to maximize label diversity.

### Step 3 — Window creation

- `create_windows_expert_sparse.py` → rslearn windows with a 3×3 center label raster
- `create_windows_dense_raster.py` → rslearn windows with full dense label raster (16×16 @ 30m upscaled 3× to 48×48 @ 10m)

Both write to the `mapbiomas_3k` rslearn dataset under `RSLEARN_EAI_ROOT`.

## Artifacts

Subsample CSVs and class distribution tables live in `datasets/mapbiomas/subsampling_artifacts/`. Model sweep results (best LR per encoder) are in `datasets/mapbiomas/model sweeps/`.

## Scripts

| Script | Purpose |
|---|---|
| `subsampling/sample_expert_points.py` | Balanced 4k subsample from 85k points |
| `subsampling/sample_dense_raster.py` | Optimize dense window centers for class diversity |
| `subsampling/visualize_sample_expert_points.py` | Visualize subsample distributions |
| `create_windows_expert_sparse.py` | Create sparse-label rslearn windows |
| `create_windows_dense_raster.py` | Create dense-label rslearn windows |
| `sanity_check.py` | Validate windows against source data + visualize |

## Note

LON/LAT columns are **swapped** in the source CSV/shapefile. All scripts handle this consistently (`longitude = row["LAT"]`, `latitude = row["LON"]`).
