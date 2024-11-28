from __future__ import annotations

import torch
from torch_geometric.data import Batch

from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from fairchem.core.models.base import HydraModel


def newforward(self, data: Batch):
    # lazily get device from input to use with amp, at least one input must be a tensor to figure out it's device
    if not self.device:
        device_from_tensors = {
            x.device.type for x in data.values() if isinstance(x, torch.Tensor)
        }
        assert (
            len(device_from_tensors) == 1
        ), f"all inputs must be on the same device, found the following devices {device_from_tensors}"
        self.device = device_from_tensors.pop()

    emb = self.backbone(data)
    # Predict all output properties for all structures in the batch for now.
    out = {}
    for k in self.output_heads:
        with torch.autocast(
            device_type=self.device, enabled=self.output_heads[k].use_amp
        ):
            if self.pass_through_head_outputs:
                out.update(self.output_heads[k](data, emb))
            else:
                out[k] = self.output_heads[k](data, emb)

    # Adapted from embedding_monkeypatch.py
    if hasattr(self, "return_embedding") and self.return_embedding:
        out["embedding"] = emb
    return out


HydraModel.forward = newforward


def embed(self, atoms) -> torch.Tensor:
    """Embed atoms using the model's backbone."""
    self.trainer._unwrapped_model.return_embedding = True

    # Adapted from embedding_monkeypatch.py
    self.calculate(atoms)

    return self.results["embedding"]


OCPCalculator.embed = embed
