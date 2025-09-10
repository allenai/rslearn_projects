"""MoE transformer for decoder trunk."""

from typing import Any

import torch
from helios.nn.moe.soft import SoftMoE
from rslearn.models.trunk import DecoderTrunkLayer


class MoETransformer(DecoderTrunkLayer):
    """Transformer for decoder trunk."""

    def __init__(
        self,
        dim: int,
        n_layers: int,
        n_heads: int,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        task_moe: bool = False,
        disable_moe: bool = False,
        num_experts: int = 16,
        num_slots: int = 256,
        expert_mult: int = 4,
        load_balance_loss_weight: float = 0.0,
    ):
        """Standard ViT-style transformer, with soft MoE.

        Since the point of the MoE layers is to deal with task-specific and task-shared
        features (and not to route specific tokens), it's probably best to use max_seq_len
        as the number of slots, and have at least one expert per task (probably more).

        Args:
            dim: dimension of the input and output
            n_layers: number of transformer blocks
            n_heads: number of attention heads
            mlp_dim: dimension of the MLP
            dropout: dropout rate
            task_moe: if specified, compute dispatch weights given the task embedding
                only, and not the token
            disable_moe: if True, disable MoE
            num_experts: number of experts in soft MoE
            num_slots: number of slots in soft MoE
            expert_mult: factor by which to multiply mlp_dim in the hidden layer of experts
            load_balance_loss_weight: weight of the load balance loss
        """
        super().__init__()
        self.disable_moe = disable_moe
        self.num_experts = num_experts
        self.num_slots = num_slots
        self.task_moe = task_moe
        self.load_balance_loss_weight = load_balance_loss_weight
        self.norm = torch.nn.LayerNorm(dim)
        self.layers = torch.nn.ModuleList([])
        for _ in range(n_layers):
            mha = torch.nn.MultiheadAttention(
                dim, n_heads, dropout=dropout, batch_first=True
            )
            if not disable_moe:
                ffn = SoftMoE(
                    dim=dim,
                    num_experts=num_experts,
                    num_slots=num_slots,
                    dropout=dropout,
                    expert_mult=expert_mult,
                )
            else:
                ffn = torch.nn.Sequential(
                    torch.nn.LayerNorm(dim),
                    torch.nn.Linear(dim, mlp_dim),
                    torch.nn.GELU(),
                    torch.nn.Linear(mlp_dim, dim),
                )
            drop = torch.nn.Dropout(dropout)
            self.layers.append(torch.nn.ModuleList([mha, ffn, drop]))

    def forward(
        self, x: torch.Tensor, task_embedding: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: input tensor of shape (batch_size, seq_len, dim)
            task_embedding: task embedding tensor of shape (batch_size, dim)

        Returns:
            dict with key "outputs" (output tensor of shape (batch_size, seq_len, dim))
            and optionally "load_balance_loss", "dispatch_weights", and "combine_weights".
        """
        # Forward pass through the transformer
        infos: list[dict[str, Any]] = []
        for mha, ffn, drop in self.layers:
            x = mha(x, x, x)[0] + x
            if not self.disable_moe:
                outs = ffn(x, weight_key=task_embedding if self.task_moe else None)
                x_ffn = outs.pop("outputs")
                infos.append(outs)
                x = drop(x_ffn + x)
            else:
                x = drop(ffn(x) + x)
        x = self.norm(x)
        outputs = {"outputs": x}

        # If using MoE, collect expert weights and auxiliary losses
        # Don't call detach because we will use this later on in the loss collation
        if not self.disable_moe:
            collated: dict[str, list[torch.Tensor]] = {
                "load_balance_loss": [],
                "dispatch_weights": [],
                "combine_weights": [],
            }
            for info in infos:
                for k, v in info.items():
                    if k == "dispatch_weights":
                        # each weight is [batch, seq_len, num_experts, num_slots]
                        # compute avg weight per token across slot/batch/expert
                        # NOTE: this is probably about the same across all tokens,
                        # assuming all tokens get looked at by a few experts
                        collated["dispatch_weights"].append(v.mean((0, 2, 3)))

                    elif k == "combine_weights":
                        # each weight is [batch, seq_len, num_experts * num_slots]
                        # compute avg weight per expert (slot group) across batch/seq
                        v = v.unflatten(-1, (self.num_experts, self.num_slots))
                        v = v.sum(-1)  # [batch, seq_len, num_experts (softmax)]
                        collated["combine_weights"].append(v.mean((0, 1)))

                    elif k == "load_balance_loss":
                        # each load balance loss per layer is a scalar
                        collated["load_balance_loss"].append(v)
            outputs.update(collated)

        return outputs

    def apply_auxiliary_losses(
        self, trunk_out: dict[str, Any], outs: dict[str, Any]
    ) -> None:
        """Apply auxiliary losses in-place.

        Just move the load balance loss to the loss dict, where it will eventually be summed.

        Args:
            trunk_out: The output of the trunk.
            outs: The output of the decoders, with key "loss_dict" containing the losses.
        """
        if "load_balance_loss" in trunk_out and self.load_balance_loss_weight > 0.0:
            total_aux_loss = torch.stack(trunk_out["load_balance_loss"]).mean()
            outs["loss_dict"]["load_balance_loss"] = (
                self.load_balance_loss_weight * total_aux_loss
            )
