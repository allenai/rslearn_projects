"""OlmoEarth model architecture."""

from .launch_finetune import launch_finetune

workflows = {
    "launch_finetune": launch_finetune,
}
