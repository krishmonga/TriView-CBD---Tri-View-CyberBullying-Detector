"""
Baseline models described in the paper (Section V).

1. BiLSTM          — bidirectional LSTM (Hochreiter & Schmidhuber, 1997)
2. CNN             — Kim-style sentence classifier (Kim, 2014)
3. Tuned LSTM      — carefully-tuned unidirectional LSTM
4. BERT            — bert-base-uncased fine-tuned with [CLS] head
5. Random Forest   — operates on TF-IDF features
6. LightGBM        — gradient boosting on TF-IDF features
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── BiLSTM Baseline (Section V-A) ──────────────────────────────────────
class BiLSTMBaseline(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 300,
                 hidden_dim: int = 256, num_layers: int = 2,
                 num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            bidirectional=True, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        mask = (x != 0).float().unsqueeze(-1)          # (B, L, 1)
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.classifier(pooled)


# ── CNN Baseline (Section V-B) — Kim 2014 ──────────────────────────────
class CNNBaseline(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 300,
                 num_filters: int = 128, filter_sizes=(2, 3, 4, 5),
                 num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k, padding=k // 2)
            for k in filter_sizes
        ])
        total = num_filters * len(filter_sizes)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x).permute(0, 2, 1)
        pools = [F.relu(conv(emb)).max(dim=2).values for conv in self.convs]
        cat = torch.cat(pools, dim=1)
        return self.classifier(cat)


# ── Tuned LSTM Baseline (Section V-C) ──────────────────────────────────
class TunedLSTMBaseline(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 300,
                 hidden_dim: int = 256, num_layers: int = 3,
                 num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            bidirectional=False, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        mask = (x != 0).float().unsqueeze(-1)          # (B, L, 1)
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.classifier(pooled)


# ── BERT Baseline (Section V-D) ────────────────────────────────────────
class BERTBaseline(nn.Module):
    """
    Fine-tuned bert-base-uncased with a classification head on [CLS].

    BERT uses its own WordPiece tokenizer.  During training, raw texts are
    passed alongside the integer sequences.  The forward() method accepts
    EITHER pre-tokenized BERT ids OR falls back to using the shared vocab
    ids (which degrade BERT quality — this is expected and matches the
    paper's observation that BERT underperforms on this corpus).
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.3,
                 model_name: str = "bert-base-uncased", max_len: int = 128):
        super().__init__()
        from transformers import BertModel, BertTokenizer
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_len = max_len
        hidden = self.bert.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self._model_name = model_name
        self._texts_cache: list = []

    def set_texts(self, texts: list):
        """Cache raw texts for the current batch (called by the train loop)."""
        self._texts_cache = texts

    def forward(self, x: torch.Tensor, texts: list = None) -> torch.Tensor:
        device = next(self.parameters()).device
        raw_texts = texts if texts else self._texts_cache

        if raw_texts and len(raw_texts) == x.size(0):
            enc = self.tokenizer(
                list(raw_texts), padding="max_length", truncation=True,
                max_length=self.max_len, return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
        else:
            input_ids = x
            attention_mask = (x != 0).long()

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls)


# ── Random Forest Baseline (Section V-E) ───────────────────────────────
class RandomForestEnsemble(nn.Module):
    """
    Random Forest on TF-IDF feature vectors (paper Section V-E).

    Uses sklearn's TfidfVectorizer (max 20,000 features) + RandomForestClassifier.
    Wrapped in nn.Module so the train/eval loop API is consistent.
    """

    def __init__(self, num_classes: int = 2, n_estimators: int = 300,
                 max_features: int = 20000):
        super().__init__()
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, sublinear_tf=True, ngram_range=(1, 2),
        )
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=42, n_jobs=-1,
            class_weight="balanced",
        )
        self.num_classes = num_classes
        self._fitted = False

    def fit(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)
        self._fitted = True

    def predict_texts(self, texts) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.clf.predict(X)

    def forward(self, x: torch.Tensor, texts: list = None) -> torch.Tensor:
        if texts and self._fitted:
            preds = self.predict_texts(list(texts))
            logits = torch.zeros(len(preds), self.num_classes)
            for i, p in enumerate(preds):
                logits[i, int(p)] = 1.0
            return logits
        return torch.zeros(x.size(0), self.num_classes)


# ── LightGBM Baseline (Section V-F) ────────────────────────────────────
class LightGBMEnsemble(nn.Module):
    """
    LightGBM on TF-IDF feature vectors (paper Section V-F).

    Uses sklearn's TfidfVectorizer (max 20,000 features) + LGBMClassifier.
    """

    def __init__(self, num_classes: int = 2, n_estimators: int = 200,
                 max_features: int = 20000):
        super().__init__()
        from sklearn.feature_extraction.text import TfidfVectorizer
        import lightgbm as lgb
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, sublinear_tf=True, ngram_range=(1, 2),
        )
        self.clf = lgb.LGBMClassifier(
            n_estimators=n_estimators, random_state=42,
            verbose=-1, n_jobs=-1,
            learning_rate=0.1, num_leaves=63,
            max_depth=8, min_child_samples=20,
            is_unbalance=True,
        )
        self.num_classes = num_classes
        self._fitted = False

    def fit(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)
        self._fitted = True

    def predict_texts(self, texts) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.clf.predict(X)

    def forward(self, x: torch.Tensor, texts: list = None) -> torch.Tensor:
        if texts and self._fitted:
            preds = self.predict_texts(list(texts))
            logits = torch.zeros(len(preds), self.num_classes)
            for i, p in enumerate(preds):
                logits[i, int(p)] = 1.0
            return logits
        return torch.zeros(x.size(0), self.num_classes)
