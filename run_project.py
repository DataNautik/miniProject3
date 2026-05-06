

"""End-to-end runner for Mini-project 3.

This script orchestrates the full project pipeline:
1. load and split MUTAG,
2. train the graph-level VAE,
3. generate 1000 Erdos-Renyi baseline graphs,
4. generate 1000 deep-model graphs,
5. compute novelty/uniqueness metrics,
6. compute graph statistics,
7. create the required 3x3 comparison figure,
8. save all report-relevant artifacts.

The script is designed to be reproducible and to keep all major steps in a
single place so the full experiment can be rerun from one entry point.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging
import sys
from dataclasses import asdict
from pathlib import Path

from cli import parse_args, get_log_level, get_device_from_args, create_train_config, create_model_config, print_config_summary
from logging_config import configure_project_logging, get_logger
from utils_io import save_json
from baseline_er import (
    DEFAULT_BASELINE_METADATA_PATH,
    DEFAULT_BASELINE_OUTPUT_PATH,
    generate_baseline_samples,
    save_baseline_metadata,
    save_baseline_samples,
)
from data import build_data_bundle
from generate import (
    DEFAULT_GENERATED_SAMPLES_PATH,
    DEFAULT_GENERATION_METADATA_PATH,
    DEFAULT_MODEL_CHECKPOINT_PATH,
    generate_from_checkpoint,
    save_generated_samples,
    save_generation_metadata,
)
from metrics import (
    DEFAULT_METRICS_PATH,
    DEFAULT_STATS_PATH,
    baseline_sample_to_networkx,
    build_training_graphs,
    compute_graph_statistics_summary,
    compute_novelty_metrics,
    generated_sample_to_networkx,
    graph_statistics_to_dict,
    novelty_metrics_to_dict,
)
from model_gvae import GraphVAEConfig
from plots import (
    DEFAULT_BINS_METADATA_PATH,
    DEFAULT_FIGURE_PATH,
    DEFAULT_TRAINING_HISTORY_FIGURE_PATH,
    save_plot_and_bins,
    save_training_history_plot,
)
from train_gvae import train_graph_vae


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SEED = 42
NUM_SAMPLES = 1000
TRAIN_NEW_MODEL = True

RUN_SUMMARY_PATH = Path("outputs") / "tables" / "run_project_summary.json"

# Initialize logging
logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------


def main(args) -> None:
    # -------------------------------------------------------------------------
    # Step 1: load data and prepare configs
    # -------------------------------------------------------------------------
    logger.info("Loading MUTAG and preparing data bundle...")
    data_bundle = build_data_bundle(seed=args.seed)

    # Create configs from arguments
    model_config = create_model_config(args)
    # Override with actual dataset values
    model_config = GraphVAEConfig(
        num_node_features=data_bundle.num_node_features,
        max_nodes=data_bundle.max_nodes,
        hidden_dim=model_config.hidden_dim,
        latent_dim=model_config.latent_dim,
        num_message_passing_steps=model_config.num_message_passing_steps,
        dropout=model_config.dropout,
        use_residual=model_config.use_residual,
        use_size_conditioning=model_config.use_size_conditioning,
    )
    
    train_config = create_train_config(args)

    # -------------------------------------------------------------------------
    # Step 2: train or reuse deep model checkpoint
    # -------------------------------------------------------------------------
    if args.train_new_model:
        logger.info("Training graph-level VAE...")
        model, history, data_bundle = train_graph_vae(
            model_config=model_config,
            train_config=train_config,
            data_bundle=data_bundle,
        )
        logger.info(f"Training completed. Logged epochs: {len(history)}")

        logger.info("Generating training history figure...")
        save_training_history_plot(history, DEFAULT_TRAINING_HISTORY_FIGURE_PATH)
    else:
        logger.info(
            f"TRAIN_NEW_MODEL=False. A checkpoint is expected at {DEFAULT_MODEL_CHECKPOINT_PATH}."
        )
        if not DEFAULT_MODEL_CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {DEFAULT_MODEL_CHECKPOINT_PATH}. "
                "Train the model first or enable TRAIN_NEW_MODEL=True."
            )

    # -------------------------------------------------------------------------
    # Step 3: generate baseline graphs
    # -------------------------------------------------------------------------
    logger.info("Generating Erdos-Renyi baseline samples...")
    baseline_samples, baseline_size_model = generate_baseline_samples(
        data_bundle=data_bundle,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    save_baseline_samples(baseline_samples, DEFAULT_BASELINE_OUTPUT_PATH)
    save_baseline_metadata(baseline_size_model, DEFAULT_BASELINE_METADATA_PATH)

    # -------------------------------------------------------------------------
    # Step 4: generate deep-model graphs
    # -------------------------------------------------------------------------
    logger.info("Generating deep-model graph samples...")
    generated_samples, deep_size_model = generate_from_checkpoint(
        model_config=model_config,
        checkpoint_path=DEFAULT_MODEL_CHECKPOINT_PATH,
        data_bundle=data_bundle,
        num_samples=args.num_samples,
        seed=args.seed,
        device=get_device_from_args(args),
    )
    save_generated_samples(generated_samples, DEFAULT_GENERATED_SAMPLES_PATH)
    save_generation_metadata(deep_size_model, NUM_SAMPLES, DEFAULT_GENERATION_METADATA_PATH)

    # -------------------------------------------------------------------------
    # Step 5: convert samples to NetworkX graphs
    # -------------------------------------------------------------------------
    logger.info("Converting graphs for evaluation...")
    training_graphs = build_training_graphs(data_bundle)
    baseline_graphs = [baseline_sample_to_networkx(sample) for sample in baseline_samples]
    deep_graphs = [generated_sample_to_networkx(sample) for sample in generated_samples]

    # -------------------------------------------------------------------------
    # Step 6: novelty / uniqueness metrics
    # -------------------------------------------------------------------------
    logger.info("Computing novelty and uniqueness metrics...")
    baseline_novelty = compute_novelty_metrics(
        generated_graphs=baseline_graphs,
        train_graphs=training_graphs,
    )
    deep_novelty = compute_novelty_metrics(
        generated_graphs=deep_graphs,
        train_graphs=training_graphs,
    )

    save_json(
        {
            "baseline": novelty_metrics_to_dict(baseline_novelty),
            "deep_model": novelty_metrics_to_dict(deep_novelty),
        },
        DEFAULT_METRICS_PATH,
    )

    # -------------------------------------------------------------------------
    # Step 7: graph statistics
    # -------------------------------------------------------------------------
    logger.info("Computing graph statistics...")
    training_stats = compute_graph_statistics_summary(training_graphs)
    baseline_stats = compute_graph_statistics_summary(baseline_graphs)
    deep_stats = compute_graph_statistics_summary(deep_graphs)

    save_json(
        {
            "training": graph_statistics_to_dict(training_stats),
            "baseline": graph_statistics_to_dict(baseline_stats),
            "deep_model": graph_statistics_to_dict(deep_stats),
        },
        DEFAULT_STATS_PATH,
    )

    # -------------------------------------------------------------------------
    # Step 8: create 3x3 comparison figure
    # -------------------------------------------------------------------------
    logger.info("Creating 3x3 histogram comparison figure...")
    save_plot_and_bins(
        training_stats=training_stats,
        baseline_stats=baseline_stats,
        deep_stats=deep_stats,
        figure_path=DEFAULT_FIGURE_PATH,
        bins_metadata_path=DEFAULT_BINS_METADATA_PATH,
    )

    # -------------------------------------------------------------------------
    # Step 9: save a compact run summary
    # -------------------------------------------------------------------------
    logger.info("Saving run summary...")
    save_json(
        {
            "seed": args.seed,
            "device": get_device_from_args(args),
            "num_samples": args.num_samples,
            "train_new_model": args.train_new_model,
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "baseline_metrics": novelty_metrics_to_dict(baseline_novelty),
            "deep_metrics": novelty_metrics_to_dict(deep_novelty),
            "artifacts": {
                "baseline_samples": str(DEFAULT_BASELINE_OUTPUT_PATH),
                "baseline_metadata": str(DEFAULT_BASELINE_METADATA_PATH),
                "deep_samples": str(DEFAULT_GENERATED_SAMPLES_PATH),
                "deep_generation_metadata": str(DEFAULT_GENERATION_METADATA_PATH),
                "evaluation_metrics": str(DEFAULT_METRICS_PATH),
                "graph_statistics": str(DEFAULT_STATS_PATH),
                "figure": str(DEFAULT_FIGURE_PATH),
                "bins_metadata": str(DEFAULT_BINS_METADATA_PATH),
                "training_history_figure": str(DEFAULT_TRAINING_HISTORY_FIGURE_PATH),
                "checkpoint": str(DEFAULT_MODEL_CHECKPOINT_PATH),
            },
        },
        RUN_SUMMARY_PATH,
    )

    logger.info("Pipeline finished successfully.")
    logger.info(f"Baseline metrics: {novelty_metrics_to_dict(baseline_novelty)}")
    logger.info(f"Deep-model metrics: {novelty_metrics_to_dict(deep_novelty)}")
    logger.info(f"Figure saved to: {DEFAULT_FIGURE_PATH}")


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Configure logging
    configure_project_logging(level=get_log_level(args), log_file=args.log_file)
    
    # Print configuration summary
    print_config_summary(args)
    
    # Run the pipeline
    try:
        main(args)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)