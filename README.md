# TriFuse — Tri-View Cyberbullying Detector

Reproducible implementation of the TriFuse multi-view framework for cyberbullying detection.

## Architecture

TriFuse fuses three complementary text representations through attention-weighted representation-level fusion:

| Branch     | Encoder                                 | Output |
| ---------- | --------------------------------------- | ------ |
| Lexical    | CNN (filters 2,3,4,5 × 64)              | 256-d  |
| Semantic   | Transformer Encoder (2 layers, 4 heads) | 256-d  |
| Structural | BiLSTM (hidden 128, 2 layers)           | 256-d  |

### Key Components

- **Frozen Pre-trained Backbone** — A frozen DistilBERT provides contextual embeddings to all three views, giving TriFuse access to pre-trained language understanding while keeping the tri-view architecture unchanged.
- **Cross-View Interaction** — Each view attends to the other two via multi-head cross-attention, enabling complementary signal extraction across views.
- **Learnable Attention Temperature** — A trainable temperature parameter prevents attention collapse to a single view and adapts during training.
- **Diversity Regularization** — An entropy-based penalty on attention weights discourages collapse and encourages balanced view usage.
- **Auxiliary Branch Heads** — Per-view classifier heads provide branch-level supervision during training only (not used at inference).
- **Two-Layer Fusion with Residual** — A deeper fusion MLP with a residual connection for richer representation-level fusion.

## Setup

```bash
pip install -r requirements.txt
```

### Datasets

Primary evaluation dataset:

- [Cyberbullying Dataset](https://www.kaggle.com/datasets/saurabhshahane/cyberbullying-dataset) from Kaggle, placed in `dataset/`

Secondary robustness dataset:

- Davidson et al. (2017) hate-speech corpus, prepared automatically into `dataset_davidson/` by running `python prepare_davidson.py`

### GloVe Embeddings

GloVe 300-d embeddings are downloaded automatically on first run. Alternatively, download `glove.6B.300d.txt` from [Stanford NLP](https://nlp.stanford.edu/projects/glove/) and place it in the project root.

### Pre-trained Backbone

TriFuse uses a frozen DistilBERT backbone (`distilbert-base-uncased`) for contextual embeddings. It is downloaded automatically from Hugging Face on first run. This requires the `transformers` library (included in `requirements.txt`).

## Usage

### Run Both Datasets (Recommended)

```bash
# Full experiment on both Shahane and Davidson datasets
# Generates results, plots, and a combined comparison log automatically
python run_all.py

# Quick test run (10 epochs)
python run_all.py --quick

# Run only one dataset
python run_all.py --datasets davidson
python run_all.py --datasets shahane
```

### Run Individual Datasets

```bash
# Full experiment: baselines + TriFuse + ablation + 5-fold CV
python main.py --mode full

# Quick test (10 epochs)
python main.py --mode full --quick

# 5-fold cross-validation only (all models)
python main.py --mode kfold --model all --k_folds 5

# Train a single model
python main.py --mode single --model trifuse

# Use a lighter transformer baseline
python main.py --mode single --model bert --bert_model_name distilbert-base-uncased

# Baselines only
python main.py --mode baseline

# Ablation study only
python main.py --mode ablation

# Davidson secondary run
python prepare_davidson.py
python main.py --mode full --data_path dataset_davidson/
```

### Available Models

| Name              | Description                                       |
| ----------------- | ------------------------------------------------- |
| `trifuse`         | Proposed TriFuse model (with pre-trained backbone) |
| `bilstm`          | BiLSTM baseline                                   |
| `cnn`             | CNN baseline (Kim 2014)                            |
| `tuned_lstm`      | Tuned unidirectional LSTM                          |
| `bert`            | Hugging Face transformer baseline                  |
| `rf`              | Random Forest on TF-IDF features                   |
| `lightgbm`        | LightGBM on TF-IDF features                        |
| `lexical_only`    | CNN branch ablation                                |
| `semantic_only`   | Transformer branch ablation                        |
| `structural_only` | BiLSTM branch ablation                             |
| `lexical_semantic` | CNN + Transformer pairwise ablation               |
| `lexical_structural` | CNN + BiLSTM pairwise ablation                 |
| `semantic_structural` | Transformer + BiLSTM pairwise ablation        |
| `no_attention`    | TriFuse with uniform (1/3) weighting               |
| `late_fusion`     | Decision-level fusion (average of branch logits)   |

## Outputs

### Single Dataset Run (`main.py`)

All results are saved in the configured output directory (default `outputs/`):

- `outputs/models/` — saved model checkpoints
- `outputs/plots/` — training curves, confusion matrices, comparison charts
- `outputs/results/` — JSON reports, LaTeX tables for the paper

### Dual Dataset Run (`run_all.py`)

When using `run_all.py`, results are organized per dataset:

- `outputs_shahane/` — Shahane dataset results, plots, and models
- `outputs_davidson/` — Davidson dataset results, plots, and models
- `combined_results_log.txt` — Side-by-side comparison log with results from both datasets
- `combined_results.json` — Machine-readable combined results for programmatic access

Each dataset output directory contains:

```
outputs_<dataset>/
├── models/          # Best model checkpoints
├── plots/           # Training curves, confusion matrices, comparisons
├── results/         # comprehensive_report.json, LaTeX tables, k-fold results
├── logs/            # Training logs
└── run.log          # Full console output from the run
```

## Project Structure

```
TriView-CBD/
├── main.py                  # Main experiment runner
├── run_all.py               # Dual-dataset runner with combined logging
├── prepare_davidson.py      # Davidson dataset preparation script
├── configs/
│   └── config.yaml          # All hyperparameters and settings
├── src/
│   ├── models.py            # TriFuse model + ablation variants
│   ├── baseline_models.py   # BiLSTM, CNN, LSTM, BERT, RF, LightGBM
│   ├── ablation_models.py   # Ablation model factory
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── attention_optimizer.py # Attention weight tracking
│   └── utils.py             # Plotting and reporting utilities
├── dataset/                 # Shahane dataset CSVs
└── dataset_davidson/        # Davidson dataset (auto-generated)
```

## Hyperparameters

See `configs/config.yaml` for all settings. Key parameters:

### General Training

| Parameter       | Value                      |
| --------------- | -------------------------- |
| Sequence length | 128                        |
| Embedding dim   | 300 (GloVe)                |
| Batch size      | 32                         |
| Learning rate   | 0.001                      |
| Optimizer       | AdamW (weight decay 0.01)  |
| Loss            | Focal Loss (γ=2.0, α=0.25) |
| Max epochs      | 100                        |
| Early stopping  | 15 epochs patience         |
| Dropout         | 0.3                        |
| Gradient clip   | 1.0                        |

### TriFuse-Specific

| Parameter                 | Value                    | Description |
| ------------------------- | ------------------------ | ----------- |
| `trifuse_use_backbone`    | `true`                   | Enable frozen pre-trained backbone |
| `trifuse_backbone`        | `distilbert-base-uncased`| Pre-trained model for contextual embeddings |
| `trifuse_lr`              | `0.001`                  | TriFuse-specific learning rate |
| `trifuse_epochs`          | `100`                    | TriFuse-specific max epochs |
| `trifuse_patience`        | `20`                     | Higher patience for convergence |
| `tri_aux_loss_weight`     | `0.25`                   | Auxiliary branch head loss weight |
| `tri_consistency_loss_weight` | `0.15`               | Branch-agreement regularizer |
| `tri_diversity_weight`    | `0.10`                   | Attention entropy regularizer |

### BERT Baseline

| Parameter         | Value                  |
| ----------------- | ---------------------- |
| `bert_model_name` | `bert-base-uncased`    |
| `bert_lr`         | `2e-5`                 |
| `bert_epochs`     | `4`                    |
| `bert_patience`   | `4`                    |

For transformer baselines, `model.bert_model_name` controls which encoder is used. Keep `bert-base-uncased` for the full baseline, or switch to a lighter model such as `distilbert-base-uncased` with `--bert_model_name`.

### Training Details

- **TriFuse** uses separate optimizer parameter groups: cross-view interaction and attention modules receive 3× the base learning rate for faster adaptation.
- **Diversity regularization** maximizes the entropy of attention weights, preventing collapse to a single dominant view.
- **Auxiliary branch heads** provide per-view supervision during training but are excluded from inference, ensuring the fusion classifier has full control.
- **Cross-view interaction** uses multi-head cross-attention where each view queries the other two, capturing complementary signals.
- **Learnable temperature** is initialized at 1.0 and clamped to [0.5, 5.0] during training.
