# SPDX-License-Identifier: Apache-2.0
"""GPU embedding extraction actor for Ray Data (NB04). Head-safe: torch and
transformers are imported lazily inside the actor (worker-only)."""

from __future__ import annotations

import numpy as np

from . import ray_common as C


class EmbeddingExtractor:
    """Stateful Ray Data actor: loads the pretrained HF decoder once per worker
    and extracts a fixed-size embedding per transaction via last-token pooling.

    Input batch (from GPUTokenizer): {"token_ids": (n, n_fields), "label": (n,)}.
    Output: {"embedding": (n, hidden), "label": (n,)} — plain numpy.
    """

    def __init__(self, model_dir: str, max_length: int = C.EMB_MAX_LENGTH, pooling: str = "last_token"):
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(model_dir).cuda().eval()
        # Embeddings only need the final hidden state, not the LM-head logits. Run
        # the base transformer (`LlamaModel`) directly: this skips the vocab
        # projection and avoids retaining every layer's hidden states. The result
        # is bit-identical to `output_hidden_states=True`[-1] (both are the final
        # RMSNorm output) but uses ~35% less peak GPU memory, so a larger inference
        # batch fits on the A10G.
        self.backbone = getattr(self.model, "model", None) or self.model.base_model
        self.max_length = max_length
        self.pooling = pooling

    def _pad(self, tok):
        n, nf = tok.shape
        L = self.max_length
        ids = np.full((n, L), C.PAD_TOKEN_ID, dtype="int64")
        ids[:, 0] = C.BOS_TOKEN_ID
        k = min(nf, L - 2)
        ids[:, 1:1 + k] = tok[:, :k]
        ids[np.arange(n), 1 + k] = C.EOS_TOKEN_ID
        return ids

    def __call__(self, batch):
        torch = self.torch
        ids = self._pad(batch["token_ids"])
        t = torch.as_tensor(ids, device="cuda")
        with torch.no_grad():
            mask = (t != C.PAD_TOKEN_ID).long()
            h = self.backbone(input_ids=t, attention_mask=mask).last_hidden_state
            if self.pooling == "last_token":
                lengths = (mask.sum(1) - 1).clamp(min=0)
                emb = h[torch.arange(t.shape[0], device=h.device), lengths]
            else:  # mean pooling
                m = mask.unsqueeze(-1).float()
                emb = (h * m).sum(1) / m.sum(1).clamp(min=1)
        # Pass through every non-token column (label + any carried raw features),
        # row-aligned with the embedding.
        out = {"embedding": emb.float().cpu().numpy()}
        for k, v in batch.items():
            if k != "token_ids":
                out[k] = v
        return out
