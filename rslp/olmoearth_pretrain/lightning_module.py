"""Training module for mixture of experts (MoE)."""

from typing import Any

from rslearn.train.lightning_module import RslearnLightningModule


class MoELightningModule(RslearnLightningModule):
    """Training with soft mixture of experts and enhanced logging."""

    def on_train_forward(
        self,
        inputs: list[dict[str, Any]],
        targets: list[dict[str, Any]],
        model_outputs: dict[str, Any],
    ) -> None:
        """Log the MOE dispatch and combine weights, plus load balancing loss.

        Args:
            inputs: The input batch.
            targets: The target batch.
            model_outputs: The output of the model, with keys "outputs" and "loss_dict", and possibly other keys.
                Expects the 'combine_weights' and 'dispatch_weights' keys to be present as well.
        """
        keys = ["combine_weights", "dispatch_weights", "load_balance_loss"]
        for k in keys:
            weights = model_outputs[k]
            if "weight" in k:
                self.log_dict(
                    {
                        f"{inputs[0]['dataset_source']}_{k}/layer{i+1}_expert{j+1}": v.item()
                        for i, layer_weights in enumerate(weights)
                        for j, v in enumerate(layer_weights)
                    },
                    batch_size=len(inputs),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
            elif "loss" in k:
                self.log_dict(
                    {
                        f"{inputs[0]['dataset_source']}_{k}/layer{i+1}": v.detach()
                        .mean()
                        .item()
                        for i, v in enumerate(weights)
                    },
                    batch_size=len(inputs),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
