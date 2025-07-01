"""
Attention Projection Layer Adaption (https://arxiv.org/pdf/2503.11335v2)
for parameter-efficient finetuning.
"""

import torch
import torch.nn.functional as F
import warnings

class SplitProjection(torch.nn.Module):

    def __init__(self, dim, r=8):
        super().__init__()
        self.dim = dim
        self.r = r

        self.indices = torch.randperm(dim)
        self.trainable_inds = self.indices[:r]
        self.frozen_inds = self.indices[r:]

        self.trainable_w = torch.nn.Parameter(torch.empty(dim, r), requires_grad=True)
        self.frozen_w = torch.nn.Parameter(torch.empty(dim, dim - r), requires_grad=False)

        self.trainable_b = torch.nn.Parameter(torch.empty(r), requires_grad=True)
        self.frozen_b = torch.nn.Parameter(torch.empty(dim - r), requires_grad=False)

    def forward(self, x):
        trainable_out = F.linear(x, self.trainable_w, self.trainable_b)
        frozen_out = F.linear(x, self.frozen_w, self.frozen_b)
        output = torch.empty(x.shape, device=x.device, dtype=trainable_out.dtype)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # Ignore scatter_ tensor warning
            output.scatter_(  # trainable part
                dim=-1,
                index=torch.tensor(self.trainable_inds, device=x.device).unsqueeze(0).unsqueeze(0).expand(x.size(0), x.size(1), -1),  # noqa
                src=trainable_out
            )
            output.scatter_(  # frozen part
                dim=-1,
                index=torch.tensor(self.frozen_inds, device=x.device).unsqueeze(0).unsqueeze(0).expand(x.size(0), x.size(1), -1),  # noqa
                src=frozen_out
            )
        
        return output

def inject_alpa(attn, r=8):
    alpa_proj = SplitProjection(attn.proj.weight.shape[0], r=r)
    with torch.no_grad():
        proj_weight = attn.proj.weight.data.clone()
        proj_bias = attn.proj.bias.data.clone()

        alpa_proj.trainable_w.data = proj_weight[alpa_proj.trainable_inds, :]
        alpa_proj.frozen_w.data = proj_weight[alpa_proj.frozen_inds, :]

        alpa_proj.trainable_b.data = proj_bias[alpa_proj.trainable_inds]
        alpa_proj.frozen_b.data = proj_bias[alpa_proj.frozen_inds]

    attn.proj = alpa_proj
    return attn
