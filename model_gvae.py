"""Graph-level variational autoencoder for Mini-project 3.

This module implements the deep generative model used in the project:
- a message-passing GNN encoder,
- a graph-level latent variable z,
- an MLP decoder that predicts adjacency logits,
- masking logic for variable-size graphs.

The model is intentionally simple and aligned with the course material on
graph-level latent VAEs. The encoder uses node features and graph structure,
while the decoder generates graph adjacency matrices only.

Design choices:
- graph-level latent variable rather than node-level latents,
- message passing encoder rather than spectral graph convolution,
- decoder predicts only the strict upper-triangle of the adjacency matrix,
  which is then expanded into a symmetric matrix.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_graph import adjacency_to_upper_triangle, upper_triangle_to_adjacency


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_HIDDEN_DIM = 64
DEFAULT_LATENT_DIM = 32
DEFAULT_NUM_MESSAGE_PASSING_STEPS = 3
DEFAULT_DROPOUT = 0.0
DEFAULT_USE_RESIDUAL = True
DEFAULT_USE_SIZE_CONDITIONING = True


# -----------------------------------------------------------------------------
# Config container
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class GraphVAEConfig:
    """Configuration for the graph-level VAE.

    Attributes:
        num_node_features: Number of input node features.
        max_nodes: Maximum number of nodes after padding.
        hidden_dim: Hidden state dimension used in the encoder and decoder.
        latent_dim: Dimension of the graph-level latent variable.
        num_message_passing_steps: Number of message-passing rounds in the encoder.
        dropout: Dropout probability used in the encoder and decoder MLPs.
        use_residual: Whether to use residual updates in message passing.
    """

    num_node_features: int
    max_nodes: int
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    latent_dim: int = DEFAULT_LATENT_DIM
    num_message_passing_steps: int = DEFAULT_NUM_MESSAGE_PASSING_STEPS
    dropout: float = DEFAULT_DROPOUT
    use_residual: bool = DEFAULT_USE_RESIDUAL
    use_size_conditioning: bool = DEFAULT_USE_SIZE_CONDITIONING

    def __post_init__(self) -> None:
        if self.num_node_features <= 0:
            raise ValueError("num_node_features must be a positive integer.")
        if self.max_nodes <= 0:
            raise ValueError("max_nodes must be a positive integer.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer.")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")
        if self.num_message_passing_steps <= 0:
            raise ValueError("num_message_passing_steps must be a positive integer.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0).")


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def count_upper_triangle_entries(num_nodes: int) -> int:
    """Return the number of strict upper-triangle entries in an adjacency matrix.

    Args:
        num_nodes: Number of nodes in the full padded adjacency.

    Returns:
        Number of upper-triangle entries.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    return num_nodes * (num_nodes - 1) // 2


def _validate_tensor_shape(tensor: torch.Tensor, expected_dims: int, name: str) -> None:
    if tensor.dim() != expected_dims:
        raise ValueError(f"{name} must have {expected_dims} dimensions, got {tensor.dim()}.")


def _validate_square_tensor(tensor: torch.Tensor, name: str) -> None:
    if tensor.dim() < 2 or tensor.shape[-1] != tensor.shape[-2]:
        raise ValueError(f"{name} must be square in its last two dimensions.")


def build_edge_mask_from_node_mask(node_mask: torch.Tensor) -> torch.Tensor:
    """Construct a strict upper-triangle mask from a node-validity mask.

    Args:
        node_mask: Tensor of shape (B, max_nodes) with 1 for valid nodes and 0 for padding.

    Returns:
        Tensor of shape (B, U), where U = max_nodes * (max_nodes - 1) // 2.
        Each entry indicates whether the corresponding upper-triangle edge belongs
        to valid nodes only.
    """
    _validate_tensor_shape(node_mask, 2, "node_mask")

    _, max_nodes = node_mask.shape
    row_idx, col_idx = torch.triu_indices(max_nodes, max_nodes, offset=1, device=node_mask.device)
    valid_edges = node_mask[:, row_idx] * node_mask[:, col_idx]
    return valid_edges
    """Construct a strict upper-triangle mask from a node-validity mask.

    Args:
        node_mask: Tensor of shape (B, max_nodes) with 1 for valid nodes and 0 for padding.

    Returns:
        Tensor of shape (B, U), where U = max_nodes * (max_nodes - 1) // 2.
        Each entry indicates whether the corresponding upper-triangle edge belongs
        to valid nodes only.
    """
    _, max_nodes = node_mask.shape
    row_idx, col_idx = torch.triu_indices(max_nodes, max_nodes, offset=1, device=node_mask.device)
    valid_edges = node_mask[:, row_idx] * node_mask[:, col_idx]
    return valid_edges


# -----------------------------------------------------------------------------
# Encoder blocks
# -----------------------------------------------------------------------------

class MessagePassingLayer(nn.Module):
    """Simple message-passing layer with sum aggregation.

    The layer performs:
        messages = A @ h
        update   = MLP([h, messages])

    Optionally, a residual connection is used.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.0, use_residual: bool = True) -> None:
        super().__init__()
        self.use_residual = use_residual

        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, node_states: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Apply one message-passing step.

        Args:
            node_states: Tensor of shape (B, N, H).
            adjacency: Tensor of shape (B, N, N).

        Returns:
            Updated node states of shape (B, N, H).
        """
        messages = torch.bmm(adjacency, node_states)
        inputs = torch.cat([node_states, messages], dim=-1)
        updates = self.update_mlp(inputs)

        if self.use_residual:
            return node_states + updates
        return updates


class GraphEncoder(nn.Module):
    """Message-passing encoder that maps a graph to posterior parameters."""

    def __init__(self, config: GraphVAEConfig) -> None:
        super().__init__()
        self.config = config

        self.input_projection = nn.Sequential(
            nn.Linear(config.num_node_features, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        self.layers = nn.ModuleList(
            [
                MessagePassingLayer(
                    hidden_dim=config.hidden_dim,
                    dropout=config.dropout,
                    use_residual=config.use_residual,
                )
                for _ in range(config.num_message_passing_steps)
            ]
        )

        self.graph_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.mu_head = nn.Linear(config.hidden_dim, config.latent_dim)
        self.logvar_head = nn.Linear(config.hidden_dim, config.latent_dim)

    def masked_mean_pool(self, node_states: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """Compute graph-level mean pooling over valid nodes only."""
        weights = node_mask.unsqueeze(-1)
        pooled_sum = (node_states * weights).sum(dim=1)
        num_valid = weights.sum(dim=1).clamp_min(1.0)
        return pooled_sum / num_valid

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of graphs into Gaussian posterior parameters.

        Args:
            node_features: Tensor of shape (B, N, F).
            adjacency: Tensor of shape (B, N, N).
            node_mask: Tensor of shape (B, N).

        Returns:
            mu: Tensor of shape (B, latent_dim).
            logvar: Tensor of shape (B, latent_dim).
        """
        _validate_tensor_shape(node_features, 3, "node_features")
        _validate_tensor_shape(adjacency, 3, "adjacency")
        _validate_tensor_shape(node_mask, 2, "node_mask")
        _validate_square_tensor(adjacency, "adjacency")

        if node_features.shape[0] != adjacency.shape[0] or node_features.shape[0] != node_mask.shape[0]:
            raise ValueError("Batch dimensions of node_features, adjacency, and node_mask must match.")
        if node_features.shape[1] != adjacency.shape[1] or node_features.shape[1] != node_mask.shape[1]:
            raise ValueError("Graph size dimensions of node_features, adjacency, and node_mask must match.")
        if node_features.shape[2] != self.config.num_node_features:
            raise ValueError(
                f"node_features last dimension must be {self.config.num_node_features}, got {node_features.shape[2]}.")

        node_states = self.input_projection(node_features)

        # Remove padded node activations at each stage.
        node_states = node_states * node_mask.unsqueeze(-1)

        for layer in self.layers:
            node_states = layer(node_states=node_states, adjacency=adjacency)
            node_states = node_states * node_mask.unsqueeze(-1)

        graph_embedding = self.masked_mean_pool(node_states=node_states, node_mask=node_mask)
        graph_embedding = self.graph_head(graph_embedding)

        mu = self.mu_head(graph_embedding)
        logvar = self.logvar_head(graph_embedding)
        return mu, logvar


# -----------------------------------------------------------------------------
# Decoder
# -----------------------------------------------------------------------------

class GraphDecoder(nn.Module):
    """MLP decoder from graph-level latent code to adjacency logits.

    Optionally accepts a per-sample node count as an additional conditioning
    signal so the decoder knows how large of a graph to produce.
    """

    def __init__(self, config: GraphVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.num_upper_entries = count_upper_triangle_entries(config.max_nodes)

        # When size conditioning is enabled, one extra scalar (normalised node
        # count) is concatenated to the latent vector before decoding.
        input_dim = config.latent_dim + (1 if config.use_size_conditioning else 0)

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, self.num_upper_entries),
        )

    def forward(self, latent: torch.Tensor, num_nodes: torch.Tensor | None = None) -> torch.Tensor:
        """Decode latent graph embeddings into upper-triangle adjacency logits.

        Args:
            latent: Tensor of shape (B, latent_dim).
            num_nodes: Optional tensor of shape (B,) with the number of valid
                nodes per graph. Used only when use_size_conditioning is True.

        Returns:
            Upper-triangle logits of shape (B, U), where
            U = max_nodes * (max_nodes - 1) // 2.
        """
        _validate_tensor_shape(latent, 2, "latent")
        if latent.shape[1] != self.config.latent_dim:
            raise ValueError(
                f"latent last dimension must be {self.config.latent_dim}, got {latent.shape[1]}.")

        if self.config.use_size_conditioning:
            if num_nodes is None:
                raise ValueError("num_nodes must be provided when use_size_conditioning is True.")
            _validate_tensor_shape(num_nodes, 1, "num_nodes")
            if num_nodes.shape[0] != latent.shape[0]:
                raise ValueError("num_nodes batch size must match latent batch size.")
            if num_nodes.min() < 1 or num_nodes.max() > self.config.max_nodes:
                raise ValueError(
                    f"num_nodes values must lie in [1, {self.config.max_nodes}], got {num_nodes.min().item()} to {num_nodes.max().item()}"
                )
            normalised = num_nodes.float().unsqueeze(-1) / self.config.max_nodes
            latent = torch.cat([latent, normalised], dim=-1)
        return self.decoder(latent)


# -----------------------------------------------------------------------------
# Main model
# -----------------------------------------------------------------------------

class GraphLevelVAE(nn.Module):
    """Graph-level VAE with message-passing encoder and MLP decoder."""

    def __init__(self, config: GraphVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = GraphEncoder(config)
        self.decoder = GraphDecoder(config)

    # -------------------------------------------------------------------------
    # Latent helpers
    # -------------------------------------------------------------------------

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from q(z|G) using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL(q(z|G) || p(z)) for a standard Gaussian prior.

        Returns:
            Tensor of shape (B,) with one KL value per graph.
        """
        return -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    # -------------------------------------------------------------------------
    # Reconstruction helpers
    # -------------------------------------------------------------------------

    def get_target_upper_triangle(self, adjacency: torch.Tensor) -> torch.Tensor:
        """Extract strict upper-triangle adjacency targets for a batch."""
        targets = [adjacency_to_upper_triangle(adj) for adj in adjacency]
        return torch.stack(targets, dim=0)

    def decode_to_adjacency_probs(
        self,
        latent: torch.Tensor,
        num_nodes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode latent variables into full adjacency probabilities.

        Args:
            latent: Tensor of shape (B, latent_dim).
            num_nodes: Optional tensor of shape (B,) passed to the decoder for
                size conditioning.

        Returns:
            Tensor of shape (B, max_nodes, max_nodes).
        """
        _validate_tensor_shape(latent, 2, "latent")
        if num_nodes is not None:
            _validate_tensor_shape(num_nodes, 1, "num_nodes")
            if num_nodes.shape[0] != latent.shape[0]:
                raise ValueError("num_nodes batch size must match latent batch size.")

        upper_logits = self.decoder(latent, num_nodes=num_nodes)
        upper_probs = torch.sigmoid(upper_logits)

        full_probs = []
        for sample_probs in upper_probs:
            adjacency = upper_triangle_to_adjacency(sample_probs, num_nodes=self.config.max_nodes)
            full_probs.append(adjacency)
        return torch.stack(full_probs, dim=0)

    def reconstruction_loss(
        self,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
        upper_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked BCE reconstruction loss on the strict upper triangle.

        Args:
            adjacency: Tensor of shape (B, N, N) containing binary targets.
            node_mask: Tensor of shape (B, N) marking valid nodes.
            upper_logits: Tensor of shape (B, U) with predicted logits.

        Returns:
            Tensor of shape (B,) with one reconstruction loss per graph.
        """
        _validate_tensor_shape(adjacency, 3, "adjacency")
        _validate_square_tensor(adjacency, "adjacency")
        _validate_tensor_shape(node_mask, 2, "node_mask")
        _validate_tensor_shape(upper_logits, 2, "upper_logits")

        if adjacency.shape[0] != node_mask.shape[0]:
            raise ValueError("Batch dimension of adjacency and node_mask must match.")

        targets = self.get_target_upper_triangle(adjacency)
        edge_mask = build_edge_mask_from_node_mask(node_mask)

        if upper_logits.shape != targets.shape:
            raise ValueError(
                f"upper_logits shape {upper_logits.shape} must match target shape {targets.shape}."
            )

        bce = F.binary_cross_entropy_with_logits(upper_logits, targets, reduction="none")
        masked_bce = bce * edge_mask

        # Normalize by number of valid edge positions to make losses comparable.
        denom = edge_mask.sum(dim=1).clamp_min(1.0)
        return masked_bce.sum(dim=1) / denom

    # -------------------------------------------------------------------------
    # Forward and loss
    # -------------------------------------------------------------------------

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run a full forward pass through the VAE.

        Returns a dictionary containing all central intermediate quantities.
        """
        mu, logvar = self.encoder(
            node_features=node_features,
            adjacency=adjacency,
            node_mask=node_mask,
        )
        latent = self.reparameterize(mu=mu, logvar=logvar)
        num_nodes = node_mask.sum(dim=1) if self.config.use_size_conditioning else None
        upper_logits = self.decoder(latent, num_nodes=num_nodes)

        return {
            "mu": mu,
            "logvar": logvar,
            "latent": latent,
            "upper_logits": upper_logits,
        }

    def loss(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
        beta: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute ELBO-based training loss.

        Args:
            node_features: Tensor of shape (B, N, F).
            adjacency: Tensor of shape (B, N, N).
            node_mask: Tensor of shape (B, N).
            beta: Weight on the KL term.

        Returns:
            total_loss: Scalar training loss.
            metrics: Dictionary with detached logging metrics.
        """
        if beta < 0.0:
            raise ValueError("beta must be non-negative.")

        outputs = self.forward(
            node_features=node_features,
            adjacency=adjacency,
            node_mask=node_mask,
        )

        recon_loss = self.reconstruction_loss(
            adjacency=adjacency,
            node_mask=node_mask,
            upper_logits=outputs["upper_logits"],
        )
        kl_loss = self.kl_divergence(mu=outputs["mu"], logvar=outputs["logvar"])

        total_loss = (recon_loss + beta * kl_loss).mean()

        metrics = {
            "loss": float(total_loss.detach().item()),
            "recon_loss": float(recon_loss.mean().detach().item()),
            "kl_loss": float(kl_loss.mean().detach().item()),
        }
        return total_loss, metrics

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def sample_latent(self, num_samples: int, device: torch.device | None = None) -> torch.Tensor:
        """Sample latent graph codes from the standard Gaussian prior."""
        if device is None:
            device = next(self.parameters()).device
        return torch.randn(num_samples, self.config.latent_dim, device=device)

    @torch.no_grad()
    def sample_adjacency_logits(self, num_samples: int, device: torch.device | None = None) -> torch.Tensor:
        """Sample decoder logits from the prior."""
        latent = self.sample_latent(num_samples=num_samples, device=device)
        return self.decoder(latent)

    @torch.no_grad()
    def sample_adjacency_probs(
        self,
        num_samples: int,
        num_nodes: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Sample full adjacency probabilities from the prior.

        Args:
            num_samples: Number of graphs to sample.
            num_nodes: Optional tensor of shape (num_samples,) with target graph
                sizes. Passed to the decoder when size conditioning is enabled.
            device: Target device for sampling.
        """
        latent = self.sample_latent(num_samples=num_samples, device=device)
        return self.decode_to_adjacency_probs(latent, num_nodes=num_nodes)


# -----------------------------------------------------------------------------
# Public builder
# -----------------------------------------------------------------------------

def build_graph_vae(config: GraphVAEConfig) -> GraphLevelVAE:
    """Build the graph-level VAE from a config object."""
    return GraphLevelVAE(config=config)


# -----------------------------------------------------------------------------
# Quick manual check
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    batch_size = 4
    max_nodes = 8
    num_features = 7

    config = GraphVAEConfig(
        num_node_features=num_features,
        max_nodes=max_nodes,
    )
    model = build_graph_vae(config)

    node_features = torch.randn(batch_size, max_nodes, num_features)
    adjacency = torch.randint(0, 2, (batch_size, max_nodes, max_nodes)).to(torch.float32)
    adjacency = torch.triu(adjacency, diagonal=1)
    adjacency = adjacency + adjacency.transpose(1, 2)
    node_mask = torch.ones(batch_size, max_nodes)

    loss, metrics = model.loss(
        node_features=node_features,
        adjacency=adjacency,
        node_mask=node_mask,
    )

    print("Graph-level VAE sanity check")
    print(f"Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")