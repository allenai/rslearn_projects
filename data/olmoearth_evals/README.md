These model and task configs are designed to work with rslp.olmoearth_evals, which
provides an adapter model to allow the inputs from all of the tasks to be consistent
with all the baseline models we want to evaluate.

The encoder freeze schedule is selected by including one of the YAMLs under
`freezes/` (e.g. `freezes/freezefor20_lrfactor1.yaml`, `freezes/frozen.yaml`). All
of them target the same module (`["model", "model", "encoder"]`), so they work
across every model. The launcher takes a `--freeze` arg to pick one.

Per-model YAMLs under `models/` are optional and only exist when a model needs
extra setup (e.g. `restore_config` to load pretrained weights).
