#!/usr/bin/env python3
"""
TriFuse — Reproducible Experiment Runner

Modes
  full       Train all models (single split) + 5-fold CV + ablation
  kfold      K-fold cross-validation only
  baseline   Train baseline models only (single split)
  ablation   Train ablation variants only (single split)
  single     Train one specific model

Usage examples
  python main.py --mode full
  python main.py --mode kfold --model all --k_folds 5
  python main.py --mode single --model trifuse
  python main.py --mode full --quick          # 10 epochs for testing
"""

import os, sys, json, argparse, warnings, yaml
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from data_loader import (
    CyberbullyingDataset, CyberbullyingTorchDataset, load_glove_embeddings,
)
from models import (TriFuseModel, TriFuseNoAttention, SingleViewClassifier,
                    PairwiseFuseModel, LateFusionModel)
from models import LexicalView, SemanticView, StructuralView
from baseline_models import (
    BiLSTMBaseline, CNNBaseline, BERTBaseline,
    TunedLSTMBaseline, RandomForestEnsemble, LightGBMEnsemble,
)
from ablation_models import create_ablation_model
from attention_optimizer import EnhancedAttentionTrainer
from utils import (
    plot_training_history, plot_confusion_matrix_with_metrics,
    plot_model_comparison, plot_kfold_results,
    create_comprehensive_report, generate_latex_tables,
)


# ── Focal Loss (Section VI-D, Eq. 11) ─────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


# ── Helpers ────────────────────────────────────────────────────────────
def load_config(path: str = "configs/config.yaml") -> dict:
    abs_path = os.path.join(ROOT, path) if not os.path.isabs(path) else path
    if os.path.exists(abs_path):
        with open(abs_path) as f:
            cfg = yaml.safe_load(f)
        print(f"Config loaded: {abs_path}")
        return cfg
    print("Config not found — using defaults.")
    return _default_config()


def _default_config():
    return {
        "system": {"device": "auto", "seed": 42, "num_workers": 4},
        "data": {
            "data_path": "dataset/", "max_seq_len": 128,
            "test_size": 0.15, "val_size": 0.15,
            "glove_path": "glove.6B.300d.txt", "vocab_size": 20000,
        },
        "model": {
            "num_classes": 2, "embed_dim": 300, "num_heads": 4,
            "cnn_filter_sizes": [2, 3, 4, 5], "cnn_num_filters": 64,
            "transformer_layers": 2, "bilstm_hidden_size": 128,
            "bilstm_num_layers": 2, "fusion_dim": 256,
            "dropout_rate": 0.3, "attention_temperature": 1.0,
            "bert_model_name": "bert-base-uncased",
            "tuned_lstm_hidden_dim": 256, "tuned_lstm_num_layers": 3,
            "rf_n_estimators": 300, "lgb_n_estimators": 200,
        },
        "training": {
            "batch_size": 32, "learning_rate": 0.001, "weight_decay": 0.01,
            "epochs": 100, "patience": 15, "focal_gamma": 2.0,
            "focal_alpha": 0.25, "gradient_clip": 1.0,
            "scheduler_T0": 10, "scheduler_T_mult": 2,
            "scheduler_eta_min": 1e-6,
            "bert_lr": 2e-5, "bert_epochs": 4, "bert_patience": 4,
        },
        "paths": {"base_output": "outputs/", "models_dir": "outputs/models/",
                   "plots_dir": "outputs/plots/", "results_dir": "outputs/results/",
                   "logs_dir": "outputs/logs/"},
    }


def _worker_seed_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    import random
    random.seed(seed)


def _reset_seed(seed):
    """Reset all RNG state — call before each fold / model build."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup(config):
    seed = config["system"]["seed"]
    _reset_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        pass
    dev = config["system"]["device"]
    device = torch.device("cuda" if (dev == "auto" and torch.cuda.is_available()) else dev if dev != "auto" else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}  "
              f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    for d in config.get("paths", {}).values():
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)
    return device


def load_data(config):
    data_path = config["data"]["data_path"]
    if not os.path.isabs(data_path):
        data_path = os.path.join(ROOT, data_path)
    ds = CyberbullyingDataset(data_path, config)
    X_tr, X_val, X_te, y_tr, y_val, y_te = ds.create_leakage_proof_splits(
        test_size=config["data"]["test_size"],
        val_size=config["data"]["val_size"],
    )
    max_len = config["data"]["max_seq_len"]
    max_vocab = config["data"].get("vocab_size", 20000)
    train_ds = CyberbullyingTorchDataset(X_tr, y_tr, max_len=max_len, max_vocab=max_vocab)
    val_ds   = CyberbullyingTorchDataset(X_val, y_val, vocab=train_ds.vocab, max_len=max_len)
    test_ds  = CyberbullyingTorchDataset(X_te, y_te, vocab=train_ds.vocab, max_len=max_len)
    vs = train_ds.get_vocab_size()
    print(f"Vocab size: {vs}")
    return train_ds, val_ds, test_ds, vs


# ── Model factory ──────────────────────────────────────────────────────
def _is_sklearn(model):
    return isinstance(model, (RandomForestEnsemble, LightGBMEnsemble))


def build_model(name: str, vocab_size: int, config: dict, device):
    mc = config["model"]
    if name == "trifuse":
        m = TriFuseModel(vocab_size, mc)
    elif name == "bilstm":
        m = BiLSTMBaseline(vocab_size, mc["embed_dim"], 256, 2,
                           mc["num_classes"], mc["dropout_rate"])
    elif name == "cnn":
        m = CNNBaseline(vocab_size, mc["embed_dim"], 128,
                        mc["cnn_filter_sizes"], mc["num_classes"], mc["dropout_rate"])
    elif name == "tuned_lstm":
        m = TunedLSTMBaseline(vocab_size, mc["embed_dim"],
                              mc.get("tuned_lstm_hidden_dim", 256),
                              mc.get("tuned_lstm_num_layers", 3),
                              mc["num_classes"], mc["dropout_rate"])
    elif name == "bert":
        m = BERTBaseline(mc["num_classes"], mc["dropout_rate"],
                         mc.get("bert_model_name", "bert-base-uncased"))
    elif name == "rf":
        m = RandomForestEnsemble(mc["num_classes"],
                                 mc.get("rf_n_estimators", 100))
    elif name == "lightgbm":
        m = LightGBMEnsemble(mc["num_classes"],
                             mc.get("lgb_n_estimators", 200))
    elif name in ("lexical_only", "semantic_only", "structural_only",
                  "lexical_semantic", "lexical_structural", "semantic_structural",
                  "no_attention", "late_fusion"):
        m = create_ablation_model(name, vocab_size, mc)
    else:
        raise ValueError(f"Unknown model: {name}")
    if not _is_sklearn(m):
        m = m.to(device)
    return m


def apply_glove(model, glove_matrix):
    if isinstance(model, BERTBaseline) or _is_sklearn(model):
        return
    if isinstance(model, (TriFuseModel, TriFuseNoAttention, LateFusionModel)):
        for view in (model.lexical_view, model.semantic_view, model.structural_view):
            view.embedding.weight.data.copy_(glove_matrix)
    elif isinstance(model, PairwiseFuseModel):
        for view in (model.view_a, model.view_b):
            view.embedding.weight.data.copy_(glove_matrix)
    elif isinstance(model, SingleViewClassifier):
        model.view.embedding.weight.data.copy_(glove_matrix)
    elif hasattr(model, "embedding"):
        model.embedding.weight.data.copy_(glove_matrix)


# ── Training loop ──────────────────────────────────────────────────────
def train_model(model, name, train_loader, val_loader, test_loader,
                config, device):
    tc = config["training"]
    mc = config["model"]

    if _is_sklearn(model):
        return _train_sklearn(model, name, train_loader, test_loader, device, mc)

    is_bert = isinstance(model, BERTBaseline)
    if is_bert:
        lr = tc.get("bert_lr", 2e-5)
        epochs = tc.get("bert_epochs", 4)
        patience = tc.get("bert_patience", 4)
    else:
        lr = tc["learning_rate"]
        epochs = tc["epochs"]
        patience = tc["patience"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=tc["weight_decay"])
    if is_bert:
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * 0.1)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy="linear",
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=tc.get("scheduler_T0", 10),
            T_mult=tc.get("scheduler_T_mult", 2),
            eta_min=tc.get("scheduler_eta_min", 1e-6),
        )
    criterion = FocalLoss(tc["focal_gamma"], tc["focal_alpha"])

    best_val, patience_ctr = 0, 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
               "learning_rates": []}

    attn_tracker = EnhancedAttentionTrainer(model)
    model_dir = os.path.join(ROOT, config["paths"].get("models_dir", "outputs/models"))
    os.makedirs(model_dir, exist_ok=True)
    best_path = os.path.join(model_dir, f"best_{name}.pth")

    print(f"\n{'='*60}\nTraining: {name}  ({epochs} epochs, patience={patience}, lr={lr})\n{'='*60}")

    for epoch in range(epochs):
        model.train()
        t_loss, t_correct, t_total = 0, 0, 0
        for seqs, labels, texts in train_loader:
            if attn_tracker._probe_batch is None and not is_bert:
                attn_tracker.set_probe_batch(seqs)
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(seqs, texts) if is_bert else model(seqs)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tc["gradient_clip"])
            optimizer.step()
            if is_bert:
                scheduler.step()
            t_loss += loss.item()
            t_correct += (logits.argmax(1) == labels).sum().item()
            t_total += labels.size(0)
        if not is_bert:
            scheduler.step()

        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for seqs, labels, texts in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                logits = model(seqs, texts) if is_bert else model(seqs)
                v_loss += criterion(logits, labels).item()
                v_correct += (logits.argmax(1) == labels).sum().item()
                v_total += labels.size(0)

        tr_acc = t_correct / t_total
        va_acc = v_correct / v_total
        history["train_loss"].append(t_loss / len(train_loader))
        history["val_loss"].append(v_loss / len(val_loader))
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["learning_rates"].append(optimizer.param_groups[0]["lr"])
        attn_tracker.record_weights()

        print(f"  Epoch {epoch+1:3d}/{epochs}  loss={t_loss/len(train_loader):.4f}  "
              f"tr_acc={tr_acc:.4f}  va_acc={va_acc:.4f}")

        if va_acc > best_val:
            best_val = va_acc
            patience_ctr = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, weights_only=True))

    results = _evaluate(model, test_loader, device, is_sklearn=False)
    results["history"] = history
    results["attention_history"] = attn_tracker.get_history()
    _print_results(name, results)
    return results


def _train_sklearn(model, name, train_loader, test_loader, device, mc):
    all_texts, all_labels = [], []
    for _, labels, texts in train_loader:
        all_texts.extend(texts)
        all_labels.extend(labels.numpy())
    model.fit(all_texts, np.array(all_labels))

    results = _evaluate(model, test_loader, device, is_sklearn=True)
    results["history"] = {}
    results["attention_history"] = []
    _print_results(name, results)
    return results


def _evaluate(model, loader, device, is_sklearn=False):
    is_bert = isinstance(model, BERTBaseline)
    if not is_sklearn:
        model.eval()
    preds_all, labels_all = [], []

    if is_sklearn:
        for _, labels, texts in loader:
            preds = model.predict_texts(list(texts))
            preds_all.extend(preds)
            labels_all.extend(labels.numpy())
    else:
        with torch.no_grad():
            for seqs, labels, texts in loader:
                seqs = seqs.to(device)
                logits = model(seqs, texts) if is_bert else model(seqs)
                preds_all.extend(logits.argmax(1).cpu().numpy())
                labels_all.extend(labels.numpy())

    return {
        "accuracy": accuracy_score(labels_all, preds_all),
        "f1_score": f1_score(labels_all, preds_all, average="weighted"),
        "precision": precision_score(labels_all, preds_all, average="weighted"),
        "recall": recall_score(labels_all, preds_all, average="weighted"),
        "predictions": preds_all,
        "labels": labels_all,
    }


def _print_results(name, res):
    print(f"\n  {name.upper()} TEST RESULTS:")
    print(f"    Accuracy:  {res['accuracy']:.4f}")
    print(f"    F1-Score:  {res['f1_score']:.4f}")
    print(f"    Precision: {res['precision']:.4f}")
    print(f"    Recall:    {res['recall']:.4f}")
    print("-" * 50)


# ── K-fold cross-validation (Section VI-B) ─────────────────────────────
def run_kfold(model_name, full_dataset, vocab_size, config, device,
              glove_matrix=None, k=5):
    """
    Paper Section VI-B: "stratified 5-fold cross-validation ...
    four folds were used for training (with a 10% held-out validation
    split for early stopping) and the remaining fold was used for testing.
    This process was repeated five times so that every instance served
    as a test sample exactly once."
    """
    print(f"\n{'='*60}\n{k}-Fold CV: {model_name}\n{'='*60}")

    labels = [full_dataset[i][1].item() for i in range(len(full_dataset))]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    fold_acc, fold_f1, fold_prec, fold_rec = [], [], [], []
    tc = config["training"]

    for fold, (train_idx, test_idx) in enumerate(skf.split(range(len(full_dataset)), labels)):
        print(f"\n  Fold {fold+1}/{k}")
        _reset_seed(config["system"]["seed"] + fold)

        test_sub = Subset(full_dataset, test_idx)
        bs = tc["batch_size"]
        te_loader = DataLoader(test_sub, batch_size=bs, shuffle=False, num_workers=0,
                               worker_init_fn=_worker_seed_fn)

        model = build_model(model_name, vocab_size, config, device)
        if glove_matrix is not None and not _is_sklearn(model):
            apply_glove(model, glove_matrix)

        if _is_sklearn(model):
            all_texts, all_labels = [], []
            for _, labs, texts in DataLoader(Subset(full_dataset, train_idx),
                                             batch_size=bs, shuffle=False, num_workers=0):
                all_texts.extend(texts)
                all_labels.extend(labs.numpy())
            model.fit(all_texts, np.array(all_labels))

            res = _evaluate(model, te_loader, device, is_sklearn=True)
        else:
            train_labels = [labels[i] for i in train_idx]
            from sklearn.model_selection import train_test_split as _split
            sub_train_idx, sub_val_idx = _split(
                range(len(train_idx)), test_size=0.1,
                stratify=train_labels, random_state=42 + fold,
            )
            actual_train_idx = [train_idx[i] for i in sub_train_idx]
            actual_val_idx = [train_idx[i] for i in sub_val_idx]

            g = torch.Generator()
            g.manual_seed(config["system"]["seed"] + fold)
            tr_loader = DataLoader(Subset(full_dataset, actual_train_idx),
                                   batch_size=bs, shuffle=True, num_workers=0,
                                   worker_init_fn=_worker_seed_fn, generator=g)
            va_loader = DataLoader(Subset(full_dataset, actual_val_idx),
                                   batch_size=bs, shuffle=False, num_workers=0,
                                   worker_init_fn=_worker_seed_fn)

            is_bert_kf = isinstance(model, BERTBaseline)
            kf_lr = tc.get("bert_lr", 2e-5) if is_bert_kf else tc["learning_rate"]
            kf_epochs = tc.get("bert_epochs", 4) if is_bert_kf else tc["epochs"]
            kf_patience = tc.get("bert_patience", 4) if is_bert_kf else tc["patience"]

            optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=kf_lr,
                                          weight_decay=tc["weight_decay"])
            if is_bert_kf:
                kf_total_steps = len(tr_loader) * kf_epochs
                kf_warmup = int(kf_total_steps * 0.1)
                kf_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=kf_lr, total_steps=kf_total_steps,
                    pct_start=kf_warmup / max(kf_total_steps, 1),
                    anneal_strategy="linear",
                )
            else:
                kf_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=tc.get("scheduler_T0", 10),
                    T_mult=tc.get("scheduler_T_mult", 2),
                    eta_min=tc.get("scheduler_eta_min", 1e-6),
                )
            criterion = FocalLoss(tc["focal_gamma"], tc["focal_alpha"])
            best_val, patience_ctr, best_state = 0, 0, None

            for epoch in range(kf_epochs):
                model.train()
                for seqs, labs, texts in tr_loader:
                    seqs, labs = seqs.to(device), labs.to(device)
                    optimizer.zero_grad()
                    logits = model(seqs, texts) if is_bert_kf else model(seqs)
                    loss = criterion(logits, labs)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), tc["gradient_clip"])
                    optimizer.step()
                    if is_bert_kf:
                        kf_scheduler.step()

                model.eval()
                vc, vt = 0, 0
                with torch.no_grad():
                    for seqs, labs, texts in va_loader:
                        seqs, labs = seqs.to(device), labs.to(device)
                        logits = model(seqs, texts) if is_bert_kf else model(seqs)
                        vc += (logits.argmax(1) == labs).sum().item()
                        vt += labs.size(0)
                va = vc / vt if vt else 0
                if not is_bert_kf:
                    kf_scheduler.step()

                if va > best_val:
                    best_val = va
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= kf_patience:
                        break

            if best_state:
                model.load_state_dict(best_state)

            res = _evaluate(model, te_loader, device, is_sklearn=False)

        fold_acc.append(res["accuracy"])
        fold_f1.append(res["f1_score"])
        fold_prec.append(res["precision"])
        fold_rec.append(res["recall"])
        print(f"    Acc={res['accuracy']:.4f}  P={res['precision']:.4f}  "
              f"R={res['recall']:.4f}  F1={res['f1_score']:.4f}")

    mean_a, std_a = np.mean(fold_acc), np.std(fold_acc)
    mean_f, std_f = np.mean(fold_f1), np.std(fold_f1)
    mean_p, std_p = np.mean(fold_prec), np.std(fold_prec)
    mean_r, std_r = np.mean(fold_rec), np.std(fold_rec)
    print(f"\n  {model_name.upper()} {k}-Fold:")
    print(f"    Acc = {mean_a:.4f} ± {std_a:.4f}")
    print(f"    P   = {mean_p:.4f} ± {std_p:.4f}")
    print(f"    R   = {mean_r:.4f} ± {std_r:.4f}")
    print(f"    F1  = {mean_f:.4f} ± {std_f:.4f}")
    print(f"  Per-fold Acc: {[f'{a:.4f}' for a in fold_acc]}")
    return {
        "model_name": model_name,
        "mean_accuracy": mean_a, "std_accuracy": std_a,
        "mean_precision": mean_p, "std_precision": std_p,
        "mean_recall": mean_r, "std_recall": std_r,
        "mean_f1_score": mean_f, "std_f1_score": std_f,
        "fold_accuracies": fold_acc, "fold_f1_scores": fold_f1,
        "fold_precisions": fold_prec, "fold_recalls": fold_rec,
    }


# ── Main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="TriFuse Cyberbullying Detection")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--mode", choices=["full", "kfold", "baseline", "ablation", "single"],
                        default="full")
    parser.add_argument("--model", default="trifuse",
                        help="Model name or 'all' / 'baselines'")
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="10-epoch quick run")
    parser.add_argument("--data_path", default=None,
                        help="Override data directory (e.g. dataset_davidson/)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_path:
        config["data"]["data_path"] = args.data_path
        print(f"*** Data path override: {args.data_path} ***")
    if args.quick:
        config["training"]["epochs"] = 10
        config["training"]["patience"] = 5
        print("*** QUICK MODE: 10 epochs ***")

    device = setup(config)
    train_ds, val_ds, test_ds, vocab_size = load_data(config)

    glove_path = config["data"].get("glove_path", "glove.6B.300d.txt")
    if not os.path.isabs(glove_path):
        glove_path = os.path.join(ROOT, glove_path)
    glove_matrix = load_glove_embeddings(glove_path, train_ds.vocab,
                                         config["model"]["embed_dim"])

    bs = config["training"]["batch_size"]
    g_main = torch.Generator()
    g_main.manual_seed(config["system"]["seed"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0,
                              worker_init_fn=_worker_seed_fn, generator=g_main)
    val_loader   = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0,
                              worker_init_fn=_worker_seed_fn)
    test_loader  = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0,
                              worker_init_fn=_worker_seed_fn)

    all_results = {}
    kfold_results = {}
    plots_dir = os.path.join(ROOT, config["paths"].get("plots_dir", "outputs/plots"))
    results_dir = os.path.join(ROOT, config["paths"].get("results_dir", "outputs/results"))

    baseline_names = ["rf", "lightgbm", "cnn", "bilstm", "tuned_lstm", "bert"]
    ablation_names = [
        "lexical_only", "semantic_only", "structural_only",
        "lexical_semantic", "lexical_structural", "semantic_structural",
        "no_attention", "late_fusion",
    ]

    # ── Single-split training ──────────────────────────────────────────
    if args.mode in ("full", "baseline", "single"):
        if args.mode == "single":
            model_list = [args.model]
        elif args.mode == "baseline":
            model_list = baseline_names
        else:
            model_list = baseline_names + ["trifuse"]

        for name in model_list:
            _reset_seed(config["system"]["seed"])
            model = build_model(name, vocab_size, config, device)
            apply_glove(model, glove_matrix)
            res = train_model(model, name, train_loader, val_loader,
                              test_loader, config, device)
            all_results[name] = res

            if res.get("history"):
                plot_training_history(res["history"], name, plots_dir)
            if res.get("predictions") and res.get("labels"):
                plot_confusion_matrix_with_metrics(
                    res["labels"], res["predictions"],
                    ["Non-Bully", "Bully"], name, plots_dir,
                )

    # ── Ablation ───────────────────────────────────────────────────────
    if args.mode in ("full", "ablation"):
        for name in ablation_names:
            _reset_seed(config["system"]["seed"])
            model = build_model(name, vocab_size, config, device)
            apply_glove(model, glove_matrix)
            res = train_model(model, name, train_loader, val_loader,
                              test_loader, config, device)
            all_results[name] = res

    # ── K-fold CV ──────────────────────────────────────────────────────
    if args.mode in ("full", "kfold"):
        full_ds = ConcatDataset([train_ds, val_ds, test_ds])

        if args.model == "all":
            kfold_models = baseline_names + ["trifuse"]
        elif args.model == "baselines":
            kfold_models = baseline_names
        elif args.mode == "full":
            kfold_models = baseline_names + ["trifuse"]
        else:
            kfold_models = [args.model]

        for name in kfold_models:
            res = run_kfold(name, full_ds, vocab_size, config, device,
                            glove_matrix, args.k_folds)
            kfold_results[name] = res

        plot_kfold_results(kfold_results, plots_dir)

    # ── Save everything ────────────────────────────────────────────────
    if all_results:
        plot_model_comparison(all_results, plots_dir)

    if all_results or kfold_results:
        report_path = os.path.join(results_dir, "comprehensive_report.json")
        create_comprehensive_report(all_results, kfold_results, report_path)
        generate_latex_tables(all_results, kfold_results, results_dir)

    # Save raw k-fold JSON
    if kfold_results:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        kf_path = os.path.join(results_dir, f"kfold_results_{ts}.json")
        serializable = {}
        for name, res in kfold_results.items():
            serializable[name] = {
                k: (float(v) if isinstance(v, (float, np.floating)) else
                    [float(x) for x in v] if isinstance(v, list) else v)
                for k, v in res.items()
            }
        with open(kf_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nK-fold results: {kf_path}")

    # ── Statistical significance tests (Wilcoxon) ───────────────────
    if kfold_results and "trifuse" in kfold_results:
        print(f"\n{'='*60}\nStatistical Significance Tests (Wilcoxon signed-rank)\n{'='*60}")
        from scipy.stats import wilcoxon
        tf_acc = kfold_results["trifuse"]["fold_accuracies"]
        sig_results = {}
        for name, res in kfold_results.items():
            if name == "trifuse":
                continue
            other_acc = res["fold_accuracies"]
            try:
                stat, p = wilcoxon(tf_acc, other_acc)
                sig = "YES (p<0.05)" if p < 0.05 else "NO"
                sig_results[name] = {"statistic": float(stat), "p_value": float(p),
                                     "significant": p < 0.05}
                print(f"  TriFuse vs {name:20s}: stat={stat:.4f}  p={p:.4f}  Significant={sig}")
            except Exception as e:
                print(f"  TriFuse vs {name:20s}: test failed ({e})")
                sig_results[name] = {"error": str(e)}

        sig_path = os.path.join(results_dir, "statistical_tests.json")
        with open(sig_path, "w") as f:
            json.dump(sig_results, f, indent=2)
        print(f"  Saved: {sig_path}")

    # ── Computational complexity analysis ──────────────────────────
    print(f"\n{'='*60}\nComputational Complexity Analysis\n{'='*60}")
    complexity = {}
    test_input = torch.randint(0, 1000, (1, config["data"]["max_seq_len"])).to(device)
    complexity_models = baseline_names + ["trifuse"] + ablation_names
    for name in complexity_models:
        try:
            model = build_model(name, vocab_size, config, device)
            if _is_sklearn(model):
                complexity[name] = {"parameters": "N/A (sklearn)", "type": "sklearn"}
                print(f"  {name:25s}: sklearn model (no neural parameters)")
                continue

            n_params = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

            import time
            model.eval()
            with torch.no_grad():
                if isinstance(model, BERTBaseline):
                    start = time.time()
                    for _ in range(10):
                        model(test_input, ["test sentence"])
                    elapsed = (time.time() - start) / 10
                else:
                    start = time.time()
                    for _ in range(100):
                        model(test_input)
                    elapsed = (time.time() - start) / 100

            throughput = 1.0 / elapsed if elapsed > 0 else 0
            complexity[name] = {
                "total_parameters": int(n_params),
                "trainable_parameters": int(trainable),
                "parameters_M": round(n_params / 1e6, 2),
                "inference_ms": round(elapsed * 1000, 2),
                "throughput_samples_per_sec": round(throughput, 1),
            }
            print(f"  {name:25s}: {n_params/1e6:7.2f}M params  "
                  f"{elapsed*1000:6.2f} ms/sample  "
                  f"{throughput:7.1f} samples/sec")
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {name:25s}: error ({e})")
            complexity[name] = {"error": str(e)}

    comp_path = os.path.join(results_dir, "complexity_analysis.json")
    with open(comp_path, "w") as f:
        json.dump(complexity, f, indent=2)
    print(f"  Saved: {comp_path}")

    print("\n" + "=" * 60)
    print("DONE.  All outputs in outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
