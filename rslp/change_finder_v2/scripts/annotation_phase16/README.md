# Phase 16: mining annotations seeded from external ASM datasets

Generates `annotations_phase16_mining_from_datasets.json` (written to
`/weka/dfive-default/rslearn-eai/datasets/change_finder/ten_year_dataset_v2_annotation_20260523/`):
1000 v2 annotation entries, 100 from each of ten public datasets of artisanal
and small-scale mining (ASM) in sub-Saharan Africa, to improve mining recall of
the `lcc_model` change-category head beyond what model-prediction-seeded phases
(11-15) surface.

Entry conventions match phase15: 128x128 window at 10 m in the local UTM zone,
group `phase16`, `time_range = pre_change +/- 3 years` when a date exists.
Window names embed the source record ID (e.g. `ipis_drc_codmine04116`,
`lames_gha_1220`, `dethier_s0349_mdg_rubi`) for traceability. Positive points
carry only `lon`/`lat` and (when the source has a date) `pre_change`;
`post_change`, `first_date_change_noticeable`, and all category fields are left
unset so prepare.py skips these points until they are fully annotated.
Entries were shuffled (seed 16016) so annotating any prefix of the file covers
all datasets.

## Datasets

| Prefix | Source | Points | pre_change | Sampling |
|---|---|---|---|---|
| `ipis_drc` | IPIS artisanal mining site visits, eastern DRC | point | visit date | 100 of the 748 sites whose first-ever visit is in 2019-2024 (long-running pre-2019 sites excluded), earliest in-window visit |
| `ipis_car` | IPIS site visits, Central African Republic | point | visit date | 100 of 360 sites, earliest in-window visit (all 2019-2021) |
| `ipis_zwe` | IPIS/ZELA site visits, Runde district, Zimbabwe | point | visit date (Feb/Mar 2019) | 100 of 317 sites |
| `usgs_copperbelt` | USGS ASM/LSM database, DRC-Zambia Copperbelt (doi:10.5066/P9LWU4FT) | **none** (1 km cells) | — (exact imagery date -> +/-3y time_range; "2019-2023" cells -> 2018-2026) | 100 of 545 `Scale=ASM` cells, window at cell centroid |
| `lames` | LAMES Ghana ASM polygons (zhu-xlab/mineseg, arXiv:2605.07740) | point in polygon | per-polygon imagery date | 100 of the 481 polygons dated >= 2019 (of 1288) |
| `amw_wa` / `amw_cb` | Africa Mining Watch Earth Index detections (West Africa / Congo Basin) | point in polygon | **none** (snapshot ~June 2026, no dates) | 74 / 26 polygons, uniform |
| `small_mines_ds` | SmallMinesDS, SW Ghana (doi:10.1109/LGRS.2025.3566356) | point | 2022-01-01 | 100 patches; pixel sampled from mask **change pixels** (0 in 2016, 1 in 2022) |
| `pasanisi_drc` | Pasanisi et al. eastern DRC ASM masks (zenodo:15257800) | point | **none** (no dates in files) | random mining pixel from 100 of 767 positive masks |
| `dethier` | Dethier et al. 2023 global river mining districts (zenodo:7699122) | **none** (district centroids) | — | 100 of 129 African districts active into 2019+ |
| `ivc` | Cote d'Ivoire ASM tiles (zenodo:20747758) | point | 2025-01-01 | random mining pixel from 100 of 608 positive tiles |

Caveats:
- Source dates are when mining was *observed* (visit or imagery date), so the
  true transition is usually earlier; `pre_change` is a starting anchor to be
  corrected during annotation.
- `dethier` centroids are for river districts spanning tens of km; the window
  is a "search near here" seed and may not itself contain mining.
- `ivc` imagery is 2025 (per the dataset's own split CSV), at the recent edge
  of the validation window.

## Reproducing

Download the sources into a data dir with this layout:

```
data_dir/
  ipis_drc.csv          # https://ipisresearch.be/wp-content/uploads/2026/05/cod_mines_curated_all_opendata_p_ipis.csv
  ipis_car.csv          # https://ipisresearch.be/wp-content/uploads/2026/05/caf_mines_curated_all_opendata_p_ipis.csv
  zwe_wb.json           # Wayback capture of the IPIS Zimbabwe WFS GeoJSON (geo.ipisresearch.be is down):
                        # http://web.archive.org/web/20211130204538id_/http://geo.ipisresearch.be/geoserver/public/ows?service=WFS&version=1.0.0&request=GetFeature&typeName=public:zwe_mines_curated_all_opendata_p_ipis&outputFormat=application%2Fjson
  usgs_copperbelt/      # unzip of Mining_Extent_Commodities_Final_reconciled_2.zip from
                        # https://www.sciencebase.gov/catalog/item/64dfd268d34e5f6cd553c2cf
  mineseg/              # git clone https://github.com/zhu-xlab/mineseg (annotations/Ghana_ASM.geojson)
  amw/                  # unzip of https://www.africaminingwatch.org/map-data/results.zip
  smallminesds/         # unzip of https://huggingface.co/datasets/ellaampy/SmallMinesDS/resolve/main/SmallMinesDS.zip
  pasanisi/             # unzip of https://zenodo.org/api/records/15257800/files/dataset.zip/content
  dethier/imports/rm_site_metadata.csv   # from imports.zip at https://zenodo.org/records/7699122
  moesm3/               # unzip of Nature Supplementary Data 1 (site KML) of doi:10.1038/s41586-023-06309-9
  ivc_tiles.zip         # https://zenodo.org/api/records/20747758/files/IVC_tiles.zip/content (read via /vsizip, no unzip)
  ivc_split.csv         # https://zenodo.org/api/records/20747758/files/ivc_total_positive_random_60_20_20_split.csv/content
```

Then:

```
python -m rslp.change_finder_v2.scripts.annotation_phase16.gen_phase16 \
    --data-dir data_dir/
python -m rslp.change_finder_v2.scripts.annotation_phase16.combine_validate \
    --data-dir data_dir/ --output annotations_phase16_mining_from_datasets.json
```

Sampling is deterministic (per-dataset seeds 16001-16010, shuffle seed 16016).
