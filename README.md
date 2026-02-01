# TriFuse: Multi-View Cyberbullying Detection with Residual Fusion

## Overview

**TriFuse** is a deep learning model that combines three complementary views (lexical, semantic, structural) with residual connections and temperature-scaled attention for robust cyberbullying detection. This repository contains the complete implementation, datasets, and evaluation code for IEEE journal submission.

## Key Features

- **Multi-View Architecture**: Lexical (CNN), Semantic (Transformer), Structural (BiLSTM)
- **Residual Connections**: Skip connections with layer normalization for improved training
- **Attention Optimization**: Temperature scaling and learnable attention mechanisms
- **Baseline Models**: BERT, Tuned LSTM, Random Forest, LightGBM for comparison
- **Ablation Studies**: Systematic component analysis
- **K-Fold Cross-Validation**: Rigorous evaluation with stratified splits
- **IEEE-Ready Reports**: LaTeX tables, CSV results, comprehensive metrics

## Project Structure

```
trifuse/
├── configs/
│   └── config.yaml              # Centralized configuration
├── dataset/
│   ├── aggression_parsed_dataset.csv
│   ├── attack_parsed_dataset.csv
│   ├── toxicity_parsed_dataset.csv
│   ├── kaggle_parsed_dataset.csv
│   ├── twitter_parsed_dataset.csv
│   ├── twitter_racism_parsed_dataset.csv
│   ├── twitter_sexism_parsed_dataset.csv
│   └── youtube_parsed_dataset.csv
├── outputs/
│   ├── models/                  # Saved model checkpoints
│   ├── results/                 # K-fold results (JSON)
│   ├── plots/                   # Visualizations
│   ├── logs/                    # Training logs
│   └── paper_tables/            # LaTeX tables for paper
├── src/
│   ├── models.py                # Core TriFuse model + ResidualBlock
│   ├── baseline_models.py       # BERT, TunedLSTM, RF, LightGBM
│   ├── ablation_models.py       # Component ablation variants
│   ├── data_loader.py           # Data loading & preprocessing
│   ├── trainer.py               # Training loop with FocalLoss
│   ├── attention_optimizer.py   # Attention regularization
│   └── utils.py                 # Visualization & reporting
├── main.py                      # Main execution script
├── requirement.txt              # Python dependencies
├── DATASET_README.md            # Dataset documentation
├── METHODOLOGY.md               # Experimental protocol
└── README.md                    # This file
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (optional, for GPU acceleration)

### Step 1: Clone & Setup
```bash
git clone <repository-url>
cd trifuse
```

### Step 2: Install Dependencies
```bash
pip install -r requirement.txt
```

Required packages:
- `torch>=2.0.0` - Neural network framework
- `transformers>=4.30.0` - Pre-trained models (BERT)
- `scikit-learn>=1.2.0` - ML utilities & ensemble methods
- `pandas>=1.5.0` - Data manipulation
- `nltk>=3.8.0` - NLP preprocessing
- `lightgbm>=4.0.0` - Gradient boosting
- `matplotlib`, `seaborn`, `plotly` - Visualization

### Step 3: Prepare Data
Ensure datasets are in `dataset/` directory:
```bash
ls dataset/
# Expected output:
# aggression_parsed_dataset.csv
# attack_parsed_dataset.csv
# toxicity_parsed_dataset.csv
# ... (8 files total)
```

See [DATASET_README.md](DATASET_README.md) for dataset sources and licensing.

## Quick Start

### 1. Full Training (All Models)
```bash
python main.py --mode full --quick
```
- Trains TriFuse + 4 baselines + 3 ablation variants
- Quick mode: 10 epochs for testing
- Time: ~5-10 minutes (GPU)

### 2. K-Fold Cross-Validation
```bash
python main.py --mode kfold --model all --k_folds 5
```
- Compares all 8 models using 5-fold stratified cross-validation
- Saves results to `outputs/kfold_results_*.json`
- Generates LaTeX table for paper in `outputs/kfold_results_table.tex`

### 3. Single Model Training
```bash
# Train specific model
python main.py --mode single --model bert
python main.py --mode single --model trifuse

# Available models: trifuse, bilstm, cnn, bert, tuned_lstm, rf, lightgbm
```

### 4. Baseline Comparison
```bash
python main.py --mode baseline --model baselines
```
- Trains only the 4 baseline models (BERT, TunedLSTM, RF, LightGBM)

## Configuration

All hyperparameters are in `configs/config.yaml`:

```yaml
# Key settings
model:
  embed_dim: 304
  num_classes: 2
  attention_temperature: 1.0      # Temperature scaling
  dropout_rate: 0.3

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  patience: 15                    # Early stopping
  use_focal_loss: true            # For imbalanced data
  gradient_clip: 1.0

data:
  max_seq_len: 128
  test_size: 0.15
  val_size: 0.15
```

## Model Architectures

### TriFuse (Proposed)
```
Input Text (128 tokens)
    ↓
┌───────┴───────┬────────────┬──────────────┐
│               │            │              │
Lexical View   Semantic     Structural     
(CNN)          (Transformer) (BiLSTM)      
  ↓              ↓            ↓
 512d           608d          304d          
    └───────┬───────┬──────────┘
            ↓
        Fusion Layer
    (ResidualBlock x2)
            ↓
    Attention (Temp=1.0)
            ↓
      Classifier
            ↓
       Logits (2-way)
```

### Baseline Models
- **BERT**: DistilBERT + classifier (3-layer)
- **TunedLSTM**: 3-layer BiLSTM + 8-head attention + pooling
- **Random Forest**: Statistical features from embeddings
- **LightGBM**: Positional + statistical features

See `src/models.py`, `src/baseline_models.py` for full implementations.

## Training & Evaluation

### K-Fold Cross-Validation Protocol
1. Combine train + validation data
2. Apply StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
3. For each fold:
   - Train on 4 folds (~80% data)
   - Validate on 1 fold (~20% data)
4. Report mean ± std of:
   - Accuracy
   - F1-Score (weighted)
   - 95% Confidence Intervals

### Loss Functions
- **PyTorch Models**: FocalLoss (γ=2.0, α=0.25) for imbalanced data
- **Ensemble Models**: Native loss (Gini for RF, LogLoss for LightGBM)

### Metrics
- Accuracy
- Precision / Recall (weighted)
- F1-Score (weighted)
- ROC-AUC
- Confusion Matrix
- 95% Confidence Intervals (from K-fold)

## Results & Output

### Generated Files

After running, check `outputs/`:

```
outputs/
├── models/
│   ├── best_TriFuse.pth
│   ├── best_BERT.pth
│   └── ...
├── results/
│   └── kfold_results_20260201_120000.json
├── plots/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── model_comparison.png
├── paper_tables/
│   ├── table1_model_comparison.tex
│   └── table2_ablation_study.tex
└── kfold_results_table.tex          # Main LaTeX table for paper
```

### Example K-Fold Results JSON

```json
{
  "trifuse": {
    "mean_accuracy": 0.8542,
    "std_accuracy": 0.0123,
    "mean_f1_score": 0.8401,
    "fold_accuracies": [0.8520, 0.8510, 0.8575, 0.8540, 0.8550]
  },
  "bert": {
    "mean_accuracy": 0.8234,
    ...
  }
}
```

### LaTeX Table for Paper

```latex
\begin{table}[htbp]
\centering
\caption{K-fold Cross Validation Results (Accuracy in \%)}
\begin{tabular}{lccc}
\toprule
\textbf{Model} & \textbf{Mean} & \textbf{Std Dev} & \textbf{95\% CI} \\
\midrule
\textbf{TriFuse (Proposed)} & 85.42 & 1.23 & (84.19, 86.65) \\
BERT & 82.34 & 2.15 & (80.19, 84.49) \\
...
```

## Reproducibility

### Fixed Parameters
- Random seed: 42
- PyTorch deterministic mode enabled
- StratifiedKFold with fixed random_state
- All hyperparameters in config.yaml

### Environment Reproduction
```bash
# Save environment
pip freeze > environment.txt

# Reproduce on another machine
pip install -r environment.txt
python main.py --mode kfold --model all --k_folds 5
```

See [METHODOLOGY.md](METHODOLOGY.md) for detailed experimental protocol.

## Citation

If you use TriFuse in your research, please cite:

```bibtex
@article{pardeep2026trifuse,
  title={TriFuse: Multi-View Cyberbullying Detection with Residual Fusion and Attention Optimization},
  author={Pardeep and others},
  journal={IEEE Transactions on ...},
  year={2026}
}
```

## Dataset Attribution

This work uses 8 public cyberbullying/toxicity datasets:
- Kaggle Cyberbullying Dataset
- Wikipedia Toxic Comments
- Twitter Hate Speech Corpus
- YouTube Comments Dataset

See [DATASET_README.md](DATASET_README.md) for full citations and licenses.

## Ethical Considerations

This research is for:
- ✅ Academic research
- ✅ Model improvement
- ✅ Better content moderation
- ❌ NOT for automated harassment or targeting

Datasets contain harmful language for research purposes only. Model predictions should not be used for autonomous decisions without human review.

## License

This project code is released under [MIT License].

Dataset licenses vary (CC0, Research Use Only). See [DATASET_README.md](DATASET_README.md).

## Contact & Support

For questions about:
- **Model**: See `src/models.py`, `src/baseline_models.py`
- **Data**: See `DATASET_README.md`
- **Experiments**: See `METHODOLOGY.md`
- **Code**: See docstrings in source files

## Version History

- **v1.0** (2026-02-01): Initial IEEE submission version
  - TriFuse model with residual connections
  - K-fold cross-validation
  - 4 baseline models for comparison
  - Comprehensive evaluation & reporting

---

**Last Updated**: 2026-02-01  
**Status**: Ready for IEEE Journal Submission
