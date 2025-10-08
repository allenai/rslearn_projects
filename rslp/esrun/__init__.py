"""esrun pipeline."""

from .esrun import esrun, finetune, one_stage, prepare_labeled_windows

workflows = {
    "esrun": esrun,
    "one_stage": one_stage,
    "prepare_labeled_windows": prepare_labeled_windows,
    "finetune": finetune,
}
