"""esrun pipeline."""

from .esrun import esrun, one_stage

workflows = {
    "esrun": esrun,
    "one_stage": one_stage,
}
