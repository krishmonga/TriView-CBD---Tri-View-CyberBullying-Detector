# TriFuse Experimental Methodology & Reproducibility Protocol

## Overview

This document specifies the exact experimental setup, hyperparameters, and evaluation protocol used for TriFuse cyberbullying detection. It ensures full reproducibility for peer review and journal publication.

---

## 1. Experimental Design

### 1.1 Data Collection & Splits

**Data Sources**:
- 8 public cyberbullying/toxicity datasets (~170K total samples)
- See DATASET_README.md for full sources and licenses

**Split Strategy**:
```
Original Data (170K+ samples)
    ↓
Load & Preprocess
    ↓
Train/Val/Test Split (Stratified)
    │
    ├─ Training: 70% (stratified)
    ├─ Validation: 15% (stratified)
    └─ Test: 15% (stratified)
    ↓
K-Fold Cross-Validation (5 folds, stratified)
```

**Implementation** (in `src/data_loader.py`):
```python
# Stratified splits with random_state=42
X_temp, X_test, y_temp, y_test = train_test_split(
    texts, labels,
    test_size=0.15,
    stratify=labels,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=val_ratio,  # val_ratio = 0.15 / 0.85
    stratify=y_temp,
    random_state=42
)

# K-Fold (in main.py)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

**Why Stratified**: Ensures balanced class distribution across all folds. Critical for imbalanced data.

### 1.2 Data Preprocessing Pipeline

**Preprocessing Steps** (in `src/data_loader.py`, `EnhancedTextPreprocessor`):

1. **Text Cleaning**:
   - Convert to lowercase
   - Remove URLs: `http\S+|www\S+|https\S+` → `[URL]`
   - Remove user mentions: `@\w+` → `[USER]`

2. **Emoji Normalization**:
   ```python
   r':\)|:-\)' → ' smile '
   r':\(|:-\(' → ' sad '
   r':D|:-D' → ' laugh '
   r'<3' → ' love '
   ```

3. **Character Handling**:
   - Remove repeated characters: `a{3,}` → `aa`
   - Remove special chars except `.,!?;:\'"-`
   - Normalize whitespace: `\s+` → ` `

4. **Length Filtering**:
   - Max sequence length: 128 tokens (pad/truncate)
   - Min text length: 3 characters
   - Remove empty texts after cleaning

**Rationale**: Cyberbullying often uses slang, repeated characters, and emojis. Normalization improves model generalization.

### 1.3 Label Definition

**Binary Classification**:
- **Class 0 (Negative)**: Non-bullying, normal comments
- **Class 1 (Positive)**: Cyberbullying, toxic, hateful, offensive

**Label Source Priority** (in updated `src/data_loader.py`):
1. Prefer `oh_label` (original label)
2. Fallback to `Annotation` column (Twitter datasets)
3. Fallback to binary integer labels (0/1)
4. Convert float scores: `≥0.5 → 1`, `<0.5 → 0`

**Class Imbalance**:
- Positive (Cyberbullying): ~35%
- Negative (Normal): ~65%

**Mitigation**:
- FocalLoss: γ=2.0, α=0.25 (focuses on hard examples)
- StratifiedKFold (maintains ratio in each fold)
- class_weight='balanced' in sklearn models

---

## 2. Model Specifications

### 2.1 TriFuse Architecture

**Input**: Tokenized text (max_length=128)
**Embedding**: 304-dimensional word embeddings

**Three Complementary Views**:

#### Lexical View (CNN)
```
Input (128 × 304)
    ↓
Conv1D: [2, 3, 4, 5]-gram filters (128 filters each = 512 total)
    ↓
BatchNorm + ReLU
    ↓
GlobalMaxPool
    ↓
Output: 512-dimensional
```

#### Semantic View (Transformer)
```
Input (128 × 304)
    ↓
Positional Encoding
    ↓
TransformerEncoder (2 layers, 8 heads, 2048 FFN)
    ↓
Mean pooling over sequence
    ↓
Output: 608-dimensional (304 × 2)
```

#### Structural View (BiLSTM)
```
Input (128 × 304)
    ↓
BiLSTM (2 layers, 256 hidden each direction = 512 total)
    ↓
Attention pooling
    ↓
Output: 304-dimensional
```

**Fusion with Residual Connections**:
```
Concatenate [512, 608, 304] = 1424-dimensional
    ↓
ResidualBlock #1 (1424 → 1024)
  - LayerNorm + GELU + Dropout(0.3)
  - Linear projection + skip connection
    ↓
ResidualBlock #2 (1024 → 512)
  - LayerNorm + GELU + Dropout(0.3)
  - Linear projection + skip connection
    ↓
Attention (Temperature Scaled)
  - Query/Key/Value projection
  - Softmax with temperature τ=1.0
  - Output: 512-dimensional
    ↓
Classifier (512 → 128 → 2)
  - Linear + ReLU + Dropout(0.3) + Linear
    ↓
Output: logits (2 classes)
```

**Why Residual Connections**: Mitigate vanishing gradients in deep fusion paths, improve gradient flow.

**Why Temperature Scaling**: τ=1.0 provides default softmax; enables hyperparameter tuning for attention sharpness.

### 2.2 Baseline Models

#### BERT Baseline
```
Input: Text tokens
    ↓
DistilBERT encoder (6 layers, pre-trained)
    ↓
[CLS] token representation
    ↓
Classifier (768 → 512 → 256 → 2)
  - Dropout(0.3) between layers
    ↓
Output: logits
```

#### Tuned LSTM Baseline
```
Input (128 × 304)
    ↓
BiLSTM (3 layers, 256 hidden)
    ↓
Multi-Head Attention (8 heads)
    ↓
Mean + Max Pooling (concatenate)
    ↓
Classifier (512 → 256 → 2)
    ↓
Output: logits
```

#### Random Forest Ensemble
```
Input: Sequence embeddings
    ↓
Feature Extraction:
  - Mean pooling per token
  - Max pooling per token
  - Min pooling per token
  - Std pooling per token
    ↓
Concatenate features → 1200-dimensional vector
    ↓
RandomForestClassifier (100 trees, max_depth=20)
    ↓
Output: class probabilities
```

#### LightGBM Ensemble
```
Input: Sequence embeddings
    ↓
Feature Engineering:
  - First token embedding (304 dims)
  - Middle token embedding (304 dims)
  - Last token embedding (304 dims)
  - Mean embedding (304 dims)
  - Max embedding (304 dims)
  - Std deviation (1 dim)
    ↓
Concatenate → 1521-dimensional vector
    ↓
LGBMClassifier (200 trees, is_unbalance=True)
    ↓
Output: class probabilities
```

### 2.3 Ablation Models

**NoAttentionTriFuse**: Remove attention mechanism, use direct pooling
**LexicalOnly**: Only CNN view
**SemanticOnly**: Only Transformer view
**StructuralOnly**: Only BiLSTM view
**TwoViewModel (3 variants)**:
- Lexical + Semantic
- Lexical + Structural
- Semantic + Structural

**Purpose**: Validate contribution of each component.

---

## 3. Training Protocol

### 3.1 Hyperparameters

```yaml
Training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  patience: 15 (early stopping)
  gradient_clip: 1.0
  
Optimizer: AdamW
Scheduler: CosineAnnealingWarmRestarts
  - warmup_steps: 500
  - T_0: 10

Loss Function: FocalLoss
  - gamma: 2.0 (focus on hard examples)
  - alpha: 0.25 (class weight)
  
Dropout: 0.3 (all layers)
```

### 3.2 Training Loop

```python
for epoch in range(epochs):
    # Training phase
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        
        logits = model(sequences)
        loss = focal_loss(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        val_loss, val_acc = evaluate(model, val_loader)
    
    scheduler.step()
    
    # Early stopping
    if val_acc > best_acc:
        best_acc = val_acc
        save_checkpoint()
    elif epoch - last_improvement > patience:
        break
```

### 3.3 Device & Reproducibility

```python
# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Ensures**: Same results across runs, even on different GPUs.

---

## 4. Evaluation Protocol

### 4.1 K-Fold Cross-Validation

```
For each of K=5 folds:
  1. Split: 80% train, 20% validation
  2. Initialize model (fresh weights)
  3. Train for ≤100 epochs (early stopping)
  4. Evaluate on validation fold
  5. Record metrics

Report:
  - Mean accuracy ± std dev
  - Mean F1-Score ± std dev
  - 95% Confidence Interval (parametric)
    CI = mean ± 1.96 * (std / sqrt(K))
```

### 4.2 Metrics

**Primary Metrics**:
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **F1-Score (weighted)**: Weighted avg of F1s per class
- **95% Confidence Interval**: From K-fold std dev

**Secondary Metrics** (per model):
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under precision-recall curve

### 4.3 Statistical Significance

**Method**: 5-fold cross-validation provides robust uncertainty estimates.

**95% CI Calculation**:
```
CI = mean ± 1.96 * (std / sqrt(5))

Example:
  mean = 85.42%, std = 1.23%
  CI = 85.42 ± 1.96 * (1.23 / 2.236)
     = 85.42 ± 1.08%
     = (84.34%, 86.50%)
```

**Model Comparison**: Non-overlapping CIs indicate statistical significance at α=0.05.

---

## 5. Implementation Details

### 5.1 Code Organization

```
src/
├── models.py
│   ├── ResidualBlock         # New: skip connections
│   ├── EnhancedLexicalView   # CNN view
│   ├── EnhancedSemanticView  # Transformer view
│   ├── EnhancedStructuralView # BiLSTM view
│   └── OptimizedTriFuseModel # Main model
│
├── baseline_models.py
│   ├── BiLSTMBaseline
│   ├── CNNBaseline
│   ├── BERTBaseline          # New
│   ├── TunedLSTMBaseline     # New
│   ├── RandomForestEnsemble  # New
│   └── LightGBMEnsemble      # New
│
├── ablation_models.py
│   ├── create_ablation_model()
│   ├── NoAttentionTriFuse
│   ├── LexicalOnlyModel
│   └── ... (8 variants total)
│
├── data_loader.py
│   ├── EnhancedTextPreprocessor
│   ├── CyberbullyingDataset
│   └── CyberbullyingTorchDataset
│
├── trainer.py
│   ├── FocalLoss             # Imbalance handling
│   └── Trainer              # Training loop
│
├── attention_optimizer.py
│   ├── AttentionOptimizer
│   ├── LearnableTemperature
│   └── EnhancedAttentionTrainer
│
└── utils.py
    ├── plot_confusion_matrix_with_metrics
    ├── plot_roc_curve
    ├── plot_precision_recall_curve
    ├── generate_latex_tables
    └── create_comprehensive_report
```

### 5.2 Configuration File

All hyperparameters in `configs/config.yaml`:
```yaml
system:
  device: auto
  seed: 42

data:
  data_path: 'dataset/'
  max_seq_len: 128
  test_size: 0.15
  val_size: 0.15

model:
  embed_dim: 304
  attention_temperature: 1.0
  dropout_rate: 0.3

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  patience: 15
  use_focal_loss: true

paths:
  models_dir: 'outputs/models/'
  results_dir: 'outputs/results/'
  paper_tables_dir: 'outputs/paper_tables/'
```

---

## 6. Output & Results

### 6.1 Results Files

After running K-fold validation:

```
outputs/
├── kfold_results_20260201_120000.json    # Main results
├── kfold_results_table.tex               # LaTeX table for paper
├── models/
│   └── best_*.pth                        # Saved model checkpoints
└── paper_tables/
    ├── table1_model_comparison.tex
    └── table2_ablation_study.tex
```

### 6.2 Results Format (JSON)

```json
{
  "trifuse": {
    "mean_accuracy": 0.8542,
    "std_accuracy": 0.0123,
    "mean_f1_score": 0.8401,
    "fold_accuracies": [0.8520, 0.8510, 0.8575, 0.8540, 0.8550],
    "fold_f1_scores": [0.8380, 0.8410, 0.8450, 0.8390, 0.8410]
  },
  "bert": { ... },
  ...
}
```

### 6.3 LaTeX Table Format

```latex
\begin{table}[htbp]
\centering
\caption{K-fold Cross Validation Results}
\label{tab:kfold_results}
\begin{tabular}{lccc}
\toprule
\textbf{Model} & \textbf{Mean Acc} & \textbf{Std Dev} & \textbf{95\% CI} \\
\midrule
\textbf{TriFuse} & 85.42\% & 1.23\% & (84.34\%, 86.50\%) \\
BERT & 82.34\% & 2.15\% & (80.19\%, 84.49\%) \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 7. Reproducibility Checklist

To reproduce exact results:

- [ ] Python 3.8+, PyTorch 2.0+, CUDA 11.8 (if GPU)
- [ ] Run: `pip install -r requirement.txt`
- [ ] Datasets in `dataset/` folder (8 CSV files)
- [ ] Run: `python main.py --mode kfold --model all --k_folds 5`
- [ ] Check `outputs/kfold_results_*.json` for results
- [ ] Compare with published results (should match within 0.1%)

**Expected Runtime**:
- 5-fold CV, all models: ~2-4 hours on NVIDIA A100
- 5-fold CV, all models: ~8-16 hours on CPU

---

## 8. Deviation & Limitations

### Known Limitations

1. **Computational Cost**: K-fold training requires 5× more computation than single train/test
2. **Dataset Overlap**: Some datasets may have similar text patterns; future work could use completely independent sources
3. **Language**: English-only; multilingual extension possible
4. **Real-Time**: Model inference requires GPU for BERT; CPU alternative with RF/LightGBM

### Potential Improvements

1. Hyperparameter search (Optuna) for better tuning
2. Ensemble of TriFuse + BERT for higher accuracy
3. Attention visualization for interpretability
4. Cross-dataset generalization testing

---

## 9. References

- Focal Loss: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
- StratifiedKFold: scikit-learn documentation
- ResidualNets: He et al., "Deep Residual Learning" (CVPR 2016)
- DistilBERT: Sanh et al., "DistilBERT, a distilled version of BERT" (2019)

---

**Version**: 1.0  
**Date**: 2026-02-01  
**Status**: Ready for IEEE Journal Review
