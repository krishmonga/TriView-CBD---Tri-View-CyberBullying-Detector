# TriFuse — Tri-View Cyberbullying Detector

Reproducible implementation of the TriFuse multi-view framework for cyberbullying detection.

## Architecture

TriFuse fuses three complementary text representations through attention-weighted representation-level fusion:


| Branch     | Encoder                                 | Output |
| ---------- | --------------------------------------- | ------ |
| Lexical    | CNN (filters 2,3,4,5 × 64)              | 256-d  |
| Semantic   | Transformer Encoder (2 layers, 4 heads) | 256-d  |
| Structural | BiLSTM (hidden 128, 2 layers)           | 256-d  |


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

## Usage

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


| Name              | Description                         |
| ----------------- | ----------------------------------- |
| `trifuse`         | Proposed TriFuse model              |
| `bilstm`          | BiLSTM baseline                     |
| `cnn`             | CNN baseline (Kim 2014)             |
| `tuned_lstm`      | Tuned unidirectional LSTM           |
| `bert`            | Hugging Face transformer baseline   |
| `rf`              | Random Forest on embedding features |
| `lightgbm`        | LightGBM on embedding features      |
| `lexical_only`    | CNN branch ablation                 |
| `semantic_only`   | Transformer branch ablation         |
| `structural_only` | BiLSTM branch ablation              |
| `no_attention`    | TriFuse with uniform weighting      |


## Outputs

All results are saved in `outputs/`:

- `outputs/models/` — saved model checkpoints
- `outputs/plots/` — training curves, confusion matrices, comparison charts
- `outputs/results/` — JSON reports, LaTeX tables for the paper

The Davidson run writes its log to `outputs/davidson_run.log`.

## Hyperparameters

See `configs/config.yaml`. Key settings:

For transformer baselines, `model.bert_model_name` controls which encoder is used. Keep `bert-base-uncased` for the full baseline, or switch to a lighter model such as `distilbert-base-uncased` when you want a compact comparison. You can also override it on the command line with `--bert_model_name`.

For TriFuse, `training.tri_aux_loss_weight` controls how strongly the branch heads are supervised during training. The default is tuned to help TriFuse learn the three views more effectively without changing the baselines.

`training.tri_consistency_loss_weight` adds a training-only branch-agreement regularizer. It does not add inference-time parameters, so the model size stays the same.


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


