"""Tests for model_gvae module."""

from __future__ import annotations

import torch
import pytest

from model_gvae import (
    GraphVAEConfig,
    GraphLevelVAE,
    MessagePassingLayer,
    GraphEncoder,
    GraphDecoder,
    build_graph_vae,
    build_edge_mask_from_node_mask,
    count_upper_triangle_entries,
)


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_count_upper_triangle_entries(self):
        """Test upper triangle counting."""
        # For n=3: 1+2=3, n*(n-1)/2 = 3*2/2 = 3
        assert count_upper_triangle_entries(3) == 3
        # For n=4: 1+2+3=6, n*(n-1)/2 = 4*3/2 = 6
        assert count_upper_triangle_entries(4) == 6
        # For n=5: 1+2+3+4=10, n*(n-1)/2 = 5*4/2 = 10
        assert count_upper_triangle_entries(5) == 10
    
    def test_build_edge_mask_from_node_mask(self):
        """Test edge mask building."""
        node_mask = torch.tensor([
            [1., 1., 0.],
            [1., 1., 1.]
        ])
        edge_mask = build_edge_mask_from_node_mask(node_mask)
        
        # Should have shape (B, num_edges)
        batch_size = 2
        num_edges = 3  # 3*2/2
        assert edge_mask.shape == (batch_size, num_edges)
        
        # First sample: only first 2 nodes valid, so only their upper triangle edge counts
        assert edge_mask[0, 0] == 1.  # (0,1) is valid
        # Second sample: all 3 nodes valid
        assert edge_mask[1, 0] == 1.  # (0,1) is valid

    def test_graph_vae_config_validates_parameters(self):
        """Test GraphVAEConfig rejects invalid configuration values."""
        with pytest.raises(ValueError, match="num_node_features must be a positive integer"):
            GraphVAEConfig(num_node_features=0, max_nodes=10)

        with pytest.raises(ValueError, match=r"dropout must be in \[0.0, 1.0\)"):
            GraphVAEConfig(num_node_features=7, max_nodes=10, dropout=1.0)
        """Test edge mask building."""
        node_mask = torch.tensor([
            [1., 1., 0.],
            [1., 1., 1.]
        ])
        edge_mask = build_edge_mask_from_node_mask(node_mask)
        
        # Should have shape (B, num_edges)
        batch_size = 2
        num_edges = 3  # 3*2/2
        assert edge_mask.shape == (batch_size, num_edges)
        
        # First sample: only first 2 nodes valid, so only their upper triangle edge counts
        assert edge_mask[0, 0] == 1.  # (0,1) is valid
        # Second sample: all 3 nodes valid
        assert edge_mask[1, 0] == 1.  # (0,1) is valid


class TestMessagePassingLayer:
    """Test MessagePassingLayer."""
    
    def test_forward_shape(self):
        """Test output shape."""
        layer = MessagePassingLayer(hidden_dim=32, use_residual=True)
        
        batch_size = 2
        num_nodes = 5
        hidden_dim = 32
        
        node_states = torch.randn(batch_size, num_nodes, hidden_dim)
        adjacency = torch.randn(batch_size, num_nodes, num_nodes)
        
        output = layer(node_states, adjacency)
        
        assert output.shape == (batch_size, num_nodes, hidden_dim)
    
    def test_residual_connection(self):
        """Test that residual connections are applied."""
        layer = MessagePassingLayer(hidden_dim=32, use_residual=True, dropout=0.0)
        
        batch_size = 1
        num_nodes = 3
        hidden_dim = 32
        
        node_states = torch.zeros(batch_size, num_nodes, hidden_dim)
        node_states[0, 0, 0] = 1.0  # Mark first state
        
        adjacency = torch.zeros(batch_size, num_nodes, num_nodes)
        
        output = layer(node_states, adjacency)
        
        # With residual, original signal should be preserved
        assert output[0, 0, 0] > 0.0  # First element should have residual


class TestGraphEncoder:
    """Test GraphEncoder."""
    
    def test_encoder_output_shape(self):
        """Test encoder output shapes."""
        config = GraphVAEConfig(
            num_node_features=7,
            max_nodes=10,
            hidden_dim=32,
            latent_dim=16,
            num_message_passing_steps=2
        )
        encoder = GraphEncoder(config)
        
        batch_size = 2
        node_features = torch.randn(batch_size, 10, 7)
        adjacency = torch.randn(batch_size, 10, 10)
        node_mask = torch.ones(batch_size, 10)
        
        mu, logvar = encoder(node_features, adjacency, node_mask)
        
        assert mu.shape == (batch_size, config.latent_dim)
        assert logvar.shape == (batch_size, config.latent_dim)
    
    def test_encoder_masking(self):
        """Test that encoder respects node masks."""
        config = GraphVAEConfig(
            num_node_features=7,
            max_nodes=10,
            hidden_dim=32,
            latent_dim=16,
        )
        encoder = GraphEncoder(config)
        
        batch_size = 1
        node_features = torch.ones(batch_size, 10, 7)
        adjacency = torch.ones(batch_size, 10, 10)
        
        # Only first 3 nodes are valid
        node_mask = torch.zeros(batch_size, 10)
        node_mask[0, :3] = 1.
        
        mu, logvar = encoder(node_features, adjacency, node_mask)
        
        # Should produce valid output
        assert not torch.isnan(mu).any()
        assert not torch.isnan(logvar).any()

    def test_encoder_rejects_non_square_adjacency(self):
        """Test encoder rejects non-square adjacency matrices."""
        config = GraphVAEConfig(
            num_node_features=7,
            max_nodes=10,
            hidden_dim=32,
            latent_dim=16,
        )
        encoder = GraphEncoder(config)

        batch_size = 1
        node_features = torch.randn(batch_size, 10, 7)
        adjacency = torch.randn(batch_size, 10, 9)
        node_mask = torch.ones(batch_size, 10)

        with pytest.raises(ValueError, match="adjacency must be square in its last two dimensions"):
            encoder(node_features, adjacency, node_mask)


class TestGraphDecoder:
    """Test GraphDecoder."""
    
    def test_decoder_output_shape(self):
        """Test decoder output shape."""
        config = GraphVAEConfig(
            num_node_features=7,
            max_nodes=10,
            latent_dim=16,
            use_size_conditioning=False
        )
        decoder = GraphDecoder(config)
        
        batch_size = 2
        latent = torch.randn(batch_size, config.latent_dim)
        
        output = decoder(latent)
        
        num_edges = count_upper_triangle_entries(10)
        assert output.shape == (batch_size, num_edges)
    
    def test_decoder_with_size_conditioning(self):
        """Test decoder with size conditioning."""
        config = GraphVAEConfig(
            num_node_features=7,
            max_nodes=10,
            latent_dim=16,
            use_size_conditioning=True
        )
        decoder = GraphDecoder(config)
        
        batch_size = 2
        latent = torch.randn(batch_size, config.latent_dim)
        num_nodes = torch.tensor([5, 7], dtype=torch.float32)
        
        output = decoder(latent, num_nodes=num_nodes)
        
        num_edges = count_upper_triangle_entries(10)
        assert output.shape == (batch_size, num_edges)

    def test_decoder_requires_num_nodes_when_conditioning(self):
        """Test decoder requires num_nodes when size conditioning is enabled."""
        config = GraphVAEConfig(
            num_node_features=7,
            max_nodes=10,
            latent_dim=16,
            use_size_conditioning=True
        )
        decoder = GraphDecoder(config)
        latent = torch.randn(2, config.latent_dim)

        with pytest.raises(ValueError, match="num_nodes must be provided when use_size_conditioning is True"):
            decoder(latent)


class TestGraphLevelVAE:
    """Test full GraphLevelVAE model."""
    
    def test_forward_pass(self):
        """Test forward pass through full model."""
        config = GraphVAEConfig(
            num_node_features=7,
            max_nodes=10,
            hidden_dim=32,
            latent_dim=16,
        )
        model = GraphLevelVAE(config)
        
        batch_size = 2
        node_features = torch.randn(batch_size, 10, 7)
        adjacency = torch.randn(batch_size, 10, 10)
        node_mask = torch.ones(batch_size, 10)
        
        output = model(node_features, adjacency, node_mask)
        
        assert "mu" in output
        assert "logvar" in output
        assert "latent" in output
        assert "upper_logits" in output
        
        assert output["mu"].shape == (batch_size, config.latent_dim)
        assert output["latent"].shape == (batch_size, config.latent_dim)
    
    def test_loss_computation(self):
        """Test loss computation."""
        config = GraphVAEConfig(
            num_node_features=7,
            max_nodes=10,
            hidden_dim=32,
            latent_dim=16,
        )
        model = GraphLevelVAE(config)
        
        batch_size = 2
        node_features = torch.randn(batch_size, 10, 7)
        adjacency = torch.ones(batch_size, 10, 10) * 0.5  # Some edges
        node_mask = torch.ones(batch_size, 10)
        
        loss, metrics = model.loss(node_features, adjacency, node_mask, beta=1.0)
        
        assert loss.item() >= 0.0
        assert "loss" in metrics
        assert "recon_loss" in metrics
        assert "kl_loss" in metrics
    
    def test_reparameterize(self):
        """Test reparameterization trick."""
        config = GraphVAEConfig(
            num_node_features=7,
            max_nodes=10,
            latent_dim=16,
        )
        model = GraphLevelVAE(config)
        
        batch_size = 2
        mu = torch.randn(batch_size, config.latent_dim)
        logvar = torch.randn(batch_size, config.latent_dim)
        
        z = model.reparameterize(mu, logvar)
        
        assert z.shape == (batch_size, config.latent_dim)
        assert not torch.isnan(z).any()
    
    def test_kl_divergence(self):
        """Test KL divergence computation."""
        config = GraphVAEConfig(
            num_node_features=7,
            max_nodes=10,
            latent_dim=16,
        )
        model = GraphLevelVAE(config)
        
        batch_size = 2
        mu = torch.zeros(batch_size, config.latent_dim)
        logvar = torch.zeros(batch_size, config.latent_dim)
        
        kl = model.kl_divergence(mu, logvar)
        
        assert kl.shape == (batch_size,)
        # KL(N(0,I) || N(0,I)) should be 0
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)


class TestBuildGraphVAE:
    """Test model building function."""
    
    def test_build_graph_vae(self):
        """Test building a GraphLevelVAE."""
        config = GraphVAEConfig(
            num_node_features=7,
            max_nodes=10,
            hidden_dim=32,
            latent_dim=16,
        )
        model = build_graph_vae(config)
        
        assert isinstance(model, GraphLevelVAE)
        assert model.config == config
        
        # Check that model is trainable
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
