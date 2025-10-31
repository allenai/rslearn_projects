These model and task configs are designed to work with rslp.olmoearth_evals, which
provides an adapter model to allow the inputs from all of the tasks to be consistent
with all the baseline models we want to evaluate.

If you want to run a frozen backbone, remove the `unfreeze_at_epoch` key in the model yaml for the backbone you want to freeze.
