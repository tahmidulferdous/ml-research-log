"""
=============================================================================
 Exponential Moving Average (EMA)
 Identical to DDPM version — keeps a smoothed copy of model weights
 for higher-quality generation.
=============================================================================
"""

import copy
import torch


class EMA:
    """
    Exponential Moving Average of model parameters.
    Keeps a smoothed copy of the model for better generation quality.

    Usage:
        ema = EMA(model, decay=0.9999)
        # After each optimizer step:
        ema.update(model)
        # For generation:
        samples = scheduler.sample(ema.shadow, ...)
    """

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        """Update EMA weights with current model weights."""
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)

    def forward(self, *args, **kwargs):
        """Forward pass through the EMA model."""
        return self.shadow(*args, **kwargs)

    def state_dict(self):
        """Return EMA model state dict for checkpointing."""
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        """Load EMA model state dict from checkpoint."""
        self.shadow.load_state_dict(state_dict)
