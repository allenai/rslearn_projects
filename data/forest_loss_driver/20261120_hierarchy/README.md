This dataset and model configuration switches the forest loss driver model from the ten
flat categories to the leaves of ACA's new three-level label hierarchy. It also adds the
Forest Loss Driver Validation 2026-09 project (annotated with the new hierarchy) and the
monocrop projects (agriculture events labeled with the crop type).

The model configuration is the same as `20260924_utm/config.yaml` except for the task and
head:

- The model has 22 outputs, one per leaf (`rslp.forest_loss_driver.hierarchy.LEAVES`).
- `HierarchicalForestLossTask` reads the `leaf_labels` property of the label feature, which
  is the set of leaves the label allows. A label from an old annotation scheme often
  maps to a non-leaf node (e.g. "agriculture") or to several leaves (e.g. the old "road"
  can be a road or a logging road).
- `HierarchicalClassificationHead` uses the marginal likelihood loss
  `-log(sum of the probabilities of the allowed leaves)`. This reduces to cross entropy
  when the label is a single leaf.
- The checkpoint is selected on `val_set_accuracy`.

Note that the deployed inference pipeline in `olmoearth_projects` still expects the flat
categories. The predictions written by `HierarchicalForestLossTask.process_output` include
the leaf, the Level 1 and Level 2 nodes, and a `legacy_category` (the flat category with
the highest summed leaf probability) to help with updating it.


Hierarchy
---------

The leaves, grouped by Level 1 / Level 2:

- Human-made
  - Agriculture: `agriculture_small_scale`, `agriculture_pasture`,
    `agriculture_soybean`, `agriculture_oil_palm`, `agriculture_rice`,
    `agriculture_other`
  - Mining: `mining_terrace`, `mining_riverine`
  - Infrastructures: `airstrip`, `road`, `infrastructure_other`
  - Forest extraction: `logging_road`, `selective_logging`
  - `burned_area`
  - `human_other` (Level 2 "Other")
- Natural: `landslide`, `blowdown`, `river_dynamics`, `flooding`, `natural_other`
  (Level 2 "Other")
- Non-forest loss: `no_change`, `already_altered`

Mennonite colonies (Level 3 "mennonites" in Studio) are not a leaf. ACA uses Level 4 to
record extra information like this. Mennonite colonies are treated as large/medium
agriculture, i.e. any agriculture leaf except `agriculture_small_scale`.


Label Mapping
-------------

The mappings are in `rslp/forest_loss_driver/hierarchy.py`. Each Studio label maps to a set
of allowed leaves:

| Source | Label | Allowed leaves |
| --- | --- | --- |
| All flat / old schemes | agriculture, agriculture-generic | all agriculture leaves |
| | mining, Mining | `mining_terrace`, `mining_riverine` |
| | road | `road`, `logging_road` |
| | logging | `selective_logging`, `logging_road` |
| | burned, Burned_areas_Fire | `burned_area`, `natural_other` (the cause is unknown) |
| | river, flood, Flooded_rivers | `river_dynamics`, `flooding` |
| | hurricane, Windthrows | `blowdown` |
| | none, False_positives | `no_change` |
| Peru 7 | agriculture-small, coca | `agriculture_small_scale` |
| | agriculture-rice | `agriculture_rice` |
| | agriculture-mennonite | agriculture except `agriculture_small_scale` |
| | human / natural | `human_other` / `natural_other` |
| ACA `tag_name` | Agriculture-Small | `agriculture_small_scale` |
| | Agriculture-Medium, Agriculture-Large | agriculture except `agriculture_small_scale` |
| | Pasture_praderas_ganaderia | `agriculture_pasture` |
| | Mining_in_River | `mining_riverine` |
| | Roads / Logging_roads / Airstrips | `road` / `logging_road` / `airstrip` |
| | Selective_Logging | `selective_logging` |
| | Logging_clear-cut, General_deforestation_Clearing, Anthropic_-_Unknown | `human_other` |
| | Urbanization_settlements | `infrastructure_other` |
| | Dry_areas_seasonal, Natural_-_Unknown | `natural_other` |
| Flat (Validatetest, Validation 2026-09 `validate`) | general_clearance | `human_other` |
| | other_natural_disturbance | `natural_other` |
| | already_altered | `already_altered` |
| Monocrop `monoculture_tag` | soybean, mennonites_soybean | `agriculture_soybean` |
| | oil_palm / pastures / rice | `agriculture_oil_palm` / `agriculture_pasture` / `agriculture_rice` |
| | mennonites_nonsoybean | agriculture except small-scale and soybean |
| | other_agriculture | `agriculture_other`, `agriculture_small_scale` |

Labels that only say the cause is unknown (`unknown`, `unlabeled`) are skipped.

Annotations made with the new hierarchy (the `level1`-`level3` fields in Validation
2026-09 and Monocrop Validation 2026-10) map directly: the allowed leaves are the leaves
under the deepest node given by the levels. A Level 3 value that is not under the given
Level 2 node is ignored. These fields take precedence over `validate` / `monoculture_tag`
when set.

The label feature also keeps `new_label`, the legacy flat category, so the older flat
configs still work. For the old schemes it is the same mapping as in `20260924_utm`. For
level-based labels it is derived from the leaves, and omitted when the leaves have no
single flat category (e.g. `human_other`).


Source Studio Projects
----------------------

The projects of `20260924_utm` (see its README), plus:

| Project | ID | Label field | Dataset group | Split |
| --- | --- | --- | --- | --- |
| Forest Loss Driver Validation 2026-09 | `3ba58828-52a0-485a-8a9b-0cd0adfc3d08` | levels, else `validate` | `20260929_validation` | hash of task ID; only `reviewed` tasks |
| Monocrop - Peru | `30b9bbb2-1ac9-4cf6-baa5-1b77ce79881c` | `monoculture_tag` | `monocrop_peru` | as in the monocrop classifier |
| Monocrop - Ecuador | `8188a029-fd50-4670-a6f2-243afc3e1b83` | `monoculture_tag` | `monocrop_ecuador` | as in the monocrop classifier |
| Monocrop - Bolivia | `dd3e9ecb-4060-4a9b-af71-05b4bf8ad747` | `monoculture_tag` | `monocrop_bolivia` | as in the monocrop classifier |
| Monocrop - Phase 2 | `ffe01218-87d6-4ebe-85c8-70b5515b951a` | `monoculture_tag` | `monocrop_phase2` | as in the monocrop classifier |
| Monocrop Validation 2026-10 | `9241501a-5fa6-49e1-a1aa-dafafe24d70b` | levels, else `monoculture_tag` | `monocrop_validation_202610` | as in the monocrop classifier; only `reviewed` tasks |

The monocrop split (`rslp.forest_loss_driver.monocrop_classifier.create_dataset.assign_split`)
keeps the monocrop classifier's validation events out of training here as well.

As of 2026-11-20 the dry run selects 6297 tasks (5675 train, 622 val), including 588
from Validation 2026-09 and 645 from the monocrop projects. Monocrop Validation 2026-10
has no reviewed tasks yet.


Usage
-----

The windows of the projects in `20260924_utm` are the same, so copy that dataset to keep
the materialized images. The script then rewrites their label layer (the window option
`label_version` is older than `create_dataset.LABEL_VERSION`) and creates windows for the
new tasks, and only those need to be materialized.

```
SRC=/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/20260924_utm/
DS=/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/20261120_hierarchy/
cp -r $SRC $DS
python -m rslp.forest_loss_driver.create_dataset --ds-path $DS
rslearn dataset prepare --root $DS --workers 64 --retry-max-attempts 5 --retry-backoff-seconds 5
rslearn dataset materialize --root $DS --workers 64 --ignore-errors --retry-max-attempts 5 --retry-backoff-seconds 5
rslearn model fit --config data/forest_loss_driver/20261120_hierarchy/config.yaml
```

`STUDIO_API_KEY` must be set. The script copies `config.json` from this directory (the
same as `20260924_utm/config.json`) into the dataset if it does not exist. Use `--dry-run` to only print the summary, which includes
the counts per label (as the covering nodes of the allowed leaves), per single leaf, and
per legacy category.


Metrics
-------

- `set_accuracy`: the predicted leaf is one of the allowed leaves.
- `leaf_accuracy`: exact accuracy on examples whose label is a single leaf.
- `level1_accuracy`, `level2_accuracy`: the leaf probabilities are summed into the
  Level 1 / Level 2 nodes, and the top node is compared with the label on examples
  whose label is within one node at that level.
- `legacy_accuracy`: the leaf probabilities are summed into the flat categories, and
  compared with `new_label`. This is roughly comparable to the `accuracy` of the flat
  model, but leaves without a flat category (`human_other`, `natural_other`,
  `infrastructure_other`) count as wrong.
- `leaf_confusion_matrix`, `level2_confusion_matrix`: logged to W&B.


Caveats
-------

- Most of the old labels map to a single leaf, but some labels are coarser. The model
  learns how to split these from the examples labeled with finer leaves. For example,
  the old `road` labels are mostly roads, but nothing in them tells the model that.
- `leaf_accuracy` and `level*_accuracy` are computed on different subsets of the
  validation set. Compare them across runs, not with each other.
- The validation set now includes Validation 2026-09 and the monocrop validation
  events, so the metrics are not comparable with `20260924_utm`.
