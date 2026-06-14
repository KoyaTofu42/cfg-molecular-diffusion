import torch
import torch.nn as nn
import sys
sys.path.append('e3_diffusion_for_molecules')
from egnn.models import EGNN

class PropertyOracle(nn.Module):
    """EGNN-based property predictor for evaluating generated molecules."""
    def __init__(self, in_node_nf, hidden_nf=128, n_layers=4, n_properties=3, device='cpu'):
        super().__init__()
        self.egnn = EGNN(
            in_node_nf=in_node_nf, in_edge_nf=1,
            hidden_nf=hidden_nf, device=device,
            act_fn=nn.SiLU(), n_layers=n_layers,
            attention=True
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, n_properties)
        )
    
    def forward(self, x, h, node_mask, edge_mask):
        """
        x: [B, N, 3] coordinates
        h: [B, N, F] node features
        """
        h_out, x_out = self.egnn(h, x, edge_mask, node_mask)
        # Masked mean pooling
        h_pooled = (h_out * node_mask).sum(dim=1) / node_mask.sum(dim=1)
        properties = self.readout(h_pooled)  # [B, n_properties]
        return properties
