# Graph Generation with Graph-Level VAE (Mini-Project 3)

This project implements a graph-level variational autoencoder (GVAE) for molecular graph generation, with comprehensive evaluation against an Erdős-Rényi baseline using the MUTAG dataset.

## 🎯 Project Overview

**Objective:** Train a deep generative model to generate novel, realistic molecular graphs and compare its performance against a statistical baseline.

**Key Components:**
- **Graph-Level VAE:** Message-passing encoder → latent variables → MLP decoder
- **Baseline:** Erdős-Rényi graphs sampled to match training distribution
- **Evaluation:** Novelty, uniqueness, and graph statistics (degree, clustering, centrality)
- **Dataset:** MUTAG (organic molecules, ~4,337 graphs)

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric 2.5+
- NetworkX 3.0+
- scikit-learn, NumPy, SciPy
- matplotlib for visualization

## 🚀 Quick Start

### Installation

```bash
# Clone repository and navigate to project
cd /path/to/miniProject3

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Full Pipeline

```bash
# Run the complete end-to-end pipeline
python run_project.py

# Or with custom configuration
python run_project.py --train-new-model --epochs 200 --batch-size 32 --device cuda
```

### Running Individual Components

```python
# Train only (with fresh model)
from train_gvae import train_graph_vae
from model_gvae import GraphVAEConfig

config = GraphVAEConfig(
    num_node_features=7,
    max_nodes=28,
    hidden_dim=64,
    latent_dim=32,
    num_message_passing_steps=3
)
model, history, data_bundle = train_graph_vae(config)

# Generate graphs
from generate import generate_graph_samples
samples = generate_graph_samples(model, size_model, num_samples=1000)

# Evaluate metrics
from metrics import compute_novelty_metrics, compute_graph_statistics_summary
novelty = compute_novelty_metrics(generated_graphs, train_graphs)
stats = compute_graph_statistics_summary(generated_graphs)
```

## 📁 Project Structure

```
miniProject3/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── run_project.py                         # End-to-end pipeline runner
│
├── Core Modules
├── model_gvae.py                          # Graph-level VAE architecture
├── train_gvae.py                          # Training routines and optimizers
├── generate.py                            # Graph sampling and generation
├── data.py                                # Data loading and preprocessing
├── baseline_er.py                         # Erdős-Rényi baseline
│
├── Evaluation & Utilities
├── metrics.py                             # Novelty, uniqueness, statistics
├── plots.py                               # Visualization (histograms)
├── utils_graph.py                         # Graph processing utilities
├── utils_io.py                            # I/O and device utilities
│
├── Output Directories
├── data/                                  # MUTAG dataset (auto-downloaded)
├── outputs/
│   ├── checkpoints/                       # Trained model weights
│   ├── samples/                           # Generated graph samples
│   ├── figures/                           # Visualization outputs
│   └── tables/                            # Metrics and metadata
│
└── tests/                                 # Unit and integration tests (optional)
```

## 🔧 Configuration

### Model Configuration (model_gvae.py)

```python
from model_gvae import GraphVAEConfig

config = GraphVAEConfig(
    num_node_features=7,           # Input node feature dimension
    max_nodes=28,                  # Maximum padded graph size
    hidden_dim=64,                 # Hidden layer dimension
    latent_dim=32,                 # Latent variable dimension
    num_message_passing_steps=3,   # Number of GNN layers
    dropout=0.0,                   # Dropout probability
    use_residual=True,             # Residual connections in GNN
    use_size_conditioning=True     # Size conditioning in decoder
)
```

### Training Configuration (train_gvae.py)

```python
from train_gvae import TrainConfig

config = TrainConfig(
    seed=42,
    device="cuda",
    epochs=200,
    batch_size=32,
    learning_rate=1e-3,
    weight_decay=1e-5,
    beta=1.0,                      # KL weight in ELBO
    beta_start=0.0,                # Initial KL weight (for annealing)
    beta_warmup_epochs=50,         # Warmup period for KL
    grad_clip_norm=5.0,            # Gradient clipping threshold
    use_early_stopping=True,
    patience=30                    # Early stopping patience
)
```

## 📊 Output Artifacts

After running the pipeline, check the `outputs/` directory:

| File | Content |
|------|---------|
| `checkpoints/best_gvae.pt` | Best trained model weights |
| `samples/baseline_er_samples.pt` | 1000 Erdős-Rényi baseline graphs |
| `samples/gvae_generated_samples.pt` | 1000 deep-model generated graphs |
| `figures/graph_statistics_3x3.png` | Histogram comparison (degree, clustering, centrality) |
| `tables/evaluation_metrics.json` | Novelty and uniqueness metrics |
| `tables/graph_statistics.json` | Raw graph statistics |
| `tables/gvae_train_history.json` | Training loss curves |
| `tables/gvae_train_config.json` | Model and training config |
| `tables/run_project_summary.json` | Summary of entire run |

## 📈 Key Metrics Explained

### Novelty & Uniqueness

```json
{
  "num_generated": 1000,
  "num_unique": 850,
  "num_novel": 720,
  "num_novel_and_unique": 650,
  "unique_fraction": 0.85,
  "novel_fraction": 0.72,
  "novel_and_unique_fraction": 0.65
}
```

- **Unique:** Graph is first in its isomorphism class among generated graphs
- **Novel:** Graph is not isomorphic to any training graph
- **Novel + Unique:** Both conditions satisfied

### Graph Statistics

Aggregated across all graphs in a set:
- **Degree:** Node connectivity distribution
- **Clustering Coefficient:** Local triangle density
- **Eigenvector Centrality:** Importance based on connection quality

## 🔬 Architecture Details

### Encoder (GraphEncoder)
1. Project node features to hidden dimension
2. Apply K message-passing layers with residual connections
3. Sum-aggregate to graph-level representation
4. Output μ and log-variance for Gaussian posterior q(z|G)

### Decoder (GraphDecoder)  
1. Optionally concatenate normalized node count to latent z
2. Pass through MLP to generate upper-triangle logits
3. Expand to symmetric binary adjacency matrix
4. Apply cleaning: binarize → remove self-loops → symmetrize

### Loss Function (ELBO)
```
L = E_q(z|G)[log p(A|z)] - β·KL(q(z|G) || p(z))
     └─ Reconstruction      └─ KL divergence (annealed)
```

## 🧪 Testing

Run the test suite (optional):

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_model.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

See [TESTING.md](TESTING.md) for detailed test documentation.

## 🎨 Visualization

### Training Curves
Automatically saved during training:
```python
from plots import save_training_history_plot
save_training_history_plot(history, "outputs/figures/training_history.png")
```

### Graph Statistics Histograms
Generated automatically as part of the pipeline:
```python
from plots import save_plot_and_bins
save_plot_and_bins(
    training_stats=training_stats,
    baseline_stats=baseline_stats,
    deep_stats=deep_stats,
    figure_path="outputs/figures/graph_statistics_3x3.png",
)
```

### Distributional Tests
Compare statistic distributions using KS tests:
```python
from metrics import compare_graph_statistics_distributions
results = compare_graph_statistics_distributions(training_stats, deep_stats)
```

## � Final Project Summary
The pipeline is designed to produce the final experiment artifacts needed for presentation and reporting.

### Latest run example
- **Deep model:** 1000 / 1000 generated graphs were novel and unique
- **Erdős-Rényi baseline:** 997 / 1000 graphs were unique and 1000 / 1000 were novel
- **Training:** stopped at epoch 79 with early stopping on the MUTAG validation set

### Output artifacts
- `final_report.tex` — LaTeX project report source
- `outputs/figures/graph_statistics_3x3.png` — histogram comparison figure
- `outputs/figures/training_history.png` — training curve plot
- `outputs/tables/evaluation_metrics.json` — novelty/uniqueness metrics
- `outputs/tables/graph_statistics.json` — raw statistic vectors
- `outputs/tables/run_project_summary.json` — compact experimental summary
- `outputs/checkpoints/best_gvae.pt` — best model checkpoint

### Build the final report
From the repository root:
```bash
pdflatex final_report.tex
```

## �🐛 Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size in TrainConfig
config.batch_size = 16  # Instead of 32

# Or reduce hidden dimensions in GraphVAEConfig
config.hidden_dim = 32  # Instead of 64
```

### Slow Data Loading
```python
# Data is cached after first load in data/MUTAG/processed/
# Delete this directory to force re-preprocessing
rm -rf data/MUTAG/processed/
```

### Model Not Improving
```python
# Try longer warmup for KL annealing
config.beta_warmup_epochs = 100  # Instead of 50

# Or use beta scheduling
config.beta_start = 0.0
config.beta = 0.1  # Start with lower weight
```

## 📚 Key References

- **Graph VAE:** Kipf & Welling (2018) - "Variational Graph Auto-Encoders"
- **Message Passing:** Gilmer et al. (2017) - "Neural Message Passing for Quantum Chemistry"
- **Evaluation:** De Cao & Kipf (2021) - "MolGen: An AI-driven Generative Model for Molecular Design"
- **MUTAG Dataset:** Morris et al. (2020) - "TUDataset: A collection of benchmark datasets for learning with graphs"

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{miniproject3_gvae,
  title={Graph-Level VAE for Molecular Graph Generation},
  author={Your Name},
  year={2026},
  note={Mini-Project 3}
}
```

## ✅ Checklist: Before Submission

- [ ] Run `python run_project.py` successfully
- [ ] Check outputs in `outputs/` directory
- [ ] Review `outputs/tables/run_project_summary.json`
- [ ] Verify `outputs/figures/graph_statistics_3x3.png` is generated
- [ ] Compare baseline vs deep-model metrics
- [ ] No errors in console output
- [ ] All dependencies installed correctly

## 🤝 Contributing

For improvements or bug fixes:

1. Create a feature branch: `git checkout -b feature/improvement`
2. Make changes with clear commit messages
3. Add tests if adding new functionality
4. Run full test suite: `pytest tests/ -v`
5. Verify with: `python run_project.py`
6. Submit pull request

## 📄 License

This project is provided as-is for educational purposes.

## 🆘 Support

For issues or questions:
1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Review code comments in relevant modules
3. Check docstrings: `help(function_name)`
4. Review test examples in `tests/`

---

**Last Updated:** May 2026  
**Version:** 1.0  
**Status:** Production-ready
