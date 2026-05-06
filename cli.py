"""Command-line interface for Mini-project 3.

This module provides a user-friendly CLI for running the project pipeline
with customizable configuration options.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from logging_config import configure_project_logging, get_logger
from train_gvae import TrainConfig, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE
from model_gvae import GraphVAEConfig, DEFAULT_HIDDEN_DIM, DEFAULT_LATENT_DIM
from utils_io import get_device


logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.
    
    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Mini-Project 3: Graph-Level VAE for Molecular Graph Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with defaults
  python run_project.py
  
  # Train a new model with custom config
  python run_project.py --train-new-model --epochs 300 --batch-size 16
  
  # Generate samples from existing checkpoint
  python run_project.py --no-train --num-samples 5000
  
  # Debug with verbose logging
  python run_project.py --verbose --seed 123
        """
    )
    
    # Training options
    train_group = parser.add_argument_group("Training Options")
    train_group.add_argument(
        "--train-new-model",
        action="store_true",
        default=True,
        help="Train a new model (default: True)"
    )
    train_group.add_argument(
        "--no-train",
        action="store_false",
        dest="train_new_model",
        help="Skip training and use existing checkpoint"
    )
    train_group.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})"
    )
    train_group.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Training batch size (default: {DEFAULT_BATCH_SIZE})"
    )
    train_group.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Initial learning rate (default: {DEFAULT_LEARNING_RATE})"
    )
    train_group.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="KL weight in ELBO loss (default: 1.0)"
    )
    train_group.add_argument(
        "--beta-warmup",
        type=int,
        default=50,
        help="Number of epochs to warm up beta (default: 50)"
    )
    train_group.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience in epochs (default: 30)"
    )
    
    # Model architecture options
    model_group = parser.add_argument_group("Model Architecture Options")
    model_group.add_argument(
        "--hidden-dim",
        type=int,
        default=DEFAULT_HIDDEN_DIM,
        help=f"Hidden dimension size (default: {DEFAULT_HIDDEN_DIM})"
    )
    model_group.add_argument(
        "--latent-dim",
        type=int,
        default=DEFAULT_LATENT_DIM,
        help=f"Latent dimension size (default: {DEFAULT_LATENT_DIM})"
    )
    model_group.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of message-passing layers (default: 3)"
    )
    model_group.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability (default: 0.0)"
    )
    model_group.add_argument(
        "--no-residual",
        action="store_false",
        dest="use_residual",
        help="Disable residual connections"
    )
    model_group.add_argument(
        "--no-size-conditioning",
        action="store_false",
        dest="use_size_conditioning",
        help="Disable size conditioning in decoder"
    )
    
    # Generation options
    gen_group = parser.add_argument_group("Generation Options")
    gen_group.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to generate (default: 1000)"
    )
    
    # System options
    sys_group = parser.add_argument_group("System Options")
    sys_group.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "mps", "auto"],
        default="auto",
        help="Device to use for training (default: auto-detect)"
    )
    sys_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results (default: outputs)"
    )
    output_group.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path (default: None, only console output)"
    )
    
    # Logging options
    log_group = parser.add_argument_group("Logging Options")
    log_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)"
    )
    log_group.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress most logging output (WARNING level)"
    )
    
    return parser


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.
    
    Args:
        args: List of arguments to parse. If None, uses sys.argv.
    
    Returns:
        Parsed arguments as Namespace.
    """
    parser = create_parser()
    return parser.parse_args(args)


def get_device_from_args(args: argparse.Namespace) -> str:
    """Get device string from parsed arguments.
    
    Args:
        args: Parsed arguments.
    
    Returns:
        Device string: "cuda", "mps", or "cpu".
    """
    if args.device == "auto":
        return get_device()
    return args.device


def get_log_level(args: argparse.Namespace) -> int:
    """Get logging level from parsed arguments.
    
    Args:
        args: Parsed arguments.
    
    Returns:
        Logging level (logging.DEBUG, logging.INFO, etc.).
    """
    if args.verbose:
        return logging.DEBUG
    elif args.quiet:
        return logging.WARNING
    else:
        return logging.INFO


def create_train_config(args: argparse.Namespace) -> TrainConfig:
    """Create TrainConfig from parsed arguments.
    
    Args:
        args: Parsed arguments.
    
    Returns:
        TrainConfig instance.
    """
    return TrainConfig(
        seed=args.seed,
        device=get_device_from_args(args),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta=args.beta,
        beta_warmup_epochs=args.beta_warmup,
        patience=args.patience,
        output_dir=args.output_dir,
    )


def create_model_config(args: argparse.Namespace) -> GraphVAEConfig:
    """Create GraphVAEConfig from parsed arguments.
    
    Args:
        args: Parsed arguments.
        num_node_features: Number of node features (typically from dataset).
        max_nodes: Maximum number of nodes (typically from dataset).
    
    Returns:
        GraphVAEConfig instance.
    """
    # These will be filled in by the main script with actual data
    return GraphVAEConfig(
        num_node_features=7,  # Will be overridden
        max_nodes=28,  # Will be overridden
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_message_passing_steps=args.num_layers,
        dropout=args.dropout,
        use_residual=args.use_residual,
        use_size_conditioning=args.use_size_conditioning,
    )


def print_config_summary(args: argparse.Namespace) -> None:
    """Print a summary of the parsed configuration.
    
    Args:
        args: Parsed arguments.
    """
    logger.info("=" * 70)
    logger.info("CONFIGURATION SUMMARY")
    logger.info("=" * 70)
    
    logger.info("\nTraining Configuration:")
    logger.info(f"  - Train new model: {args.train_new_model}")
    logger.info(f"  - Epochs: {args.epochs}")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - Learning rate: {args.learning_rate}")
    logger.info(f"  - Beta (KL weight): {args.beta}")
    logger.info(f"  - Early stopping patience: {args.patience}")
    
    logger.info("\nModel Architecture:")
    logger.info(f"  - Hidden dimension: {args.hidden_dim}")
    logger.info(f"  - Latent dimension: {args.latent_dim}")
    logger.info(f"  - Message-passing layers: {args.num_layers}")
    logger.info(f"  - Dropout: {args.dropout}")
    logger.info(f"  - Residual connections: {args.use_residual}")
    logger.info(f"  - Size conditioning: {args.use_size_conditioning}")
    
    logger.info("\nGeneration:")
    logger.info(f"  - Number of samples: {args.num_samples}")
    
    logger.info("\nSystem:")
    logger.info(f"  - Device: {get_device_from_args(args)}")
    logger.info(f"  - Seed: {args.seed}")
    logger.info(f"  - Output directory: {args.output_dir}")
    
    logger.info("=" * 70 + "\n")


if __name__ == "__main__":
    # Example usage
    args = parse_args()
    configure_project_logging(level=get_log_level(args), log_file=args.log_file)
    print_config_summary(args)
    logger.info("CLI arguments parsed successfully!")
