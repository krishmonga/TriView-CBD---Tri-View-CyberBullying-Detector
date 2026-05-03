"""
Data loading and preprocessing for the TriFuse experiments.

Primary dataset: Shahane Cyberbullying Dataset (Kaggle) — binary classification.
Secondary dataset: Davidson hate-speech corpus via ``--data_path dataset_davidson/``.
Preprocessing (Section III-B):
  1. Lowercase
  2. URL → [URL], @mention → [USER]
  3. Emoji normalization
  4. Character repetition reduction (3+ → 2)
  5. Special char removal (keep sentence punctuation)
  6. Whitespace normalization
  7. NLTK tokenization
  8. Vocabulary: top 20,000 tokens + PAD/UNK/BOS/EOS
  9. Sequence length: L = 128
    10. GloVe 300-d embeddings
"""

import os
import re
import subprocess
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

try:
    from nltk.tokenize import word_tokenize
    import nltk
    nltk.data.find("tokenizers/punkt_tab")
except Exception:
    import nltk
    nltk.download("punkt_tab", quiet=True)
    nltk.download("punkt", quiet=True)
    from nltk.tokenize import word_tokenize


# ── GloVe loader ───────────────────────────────────────────────────────
def load_glove_embeddings(glove_path: str, vocab: Dict[str, int],
                          embed_dim: int = 300) -> torch.FloatTensor:
    if not os.path.isfile(glove_path):
        _download_glove(glove_path)

    vocab_size = len(vocab)
    matrix = torch.zeros(vocab_size, embed_dim)
    nn.init.xavier_uniform_(matrix)
    for idx in vocab.values():
        if idx == 0:
            matrix[idx] = torch.zeros(embed_dim)

    found = 0
    print(f"Loading GloVe from {glove_path} ...")
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in vocab:
                try:
                    vec = torch.tensor([float(v) for v in parts[1:]])
                    if vec.size(0) == embed_dim:
                        matrix[vocab[word]] = vec
                        found += 1
                except ValueError:
                    continue

    pct = found / vocab_size * 100
    print(f"GloVe: {found}/{vocab_size} tokens loaded ({pct:.1f}% coverage)")
    return matrix


def _download_glove(glove_path: str):
    zip_path = "glove.6B.zip"
    url = "https://nlp.stanford.edu/data/glove.6B.zip"
    if not os.path.isfile(zip_path):
        print(f"Downloading GloVe from {url} (~862 MB) ...")
        urllib.request.urlretrieve(url, zip_path, _dl_progress)
        print()
    target = os.path.basename(glove_path)
    print(f"Extracting {target} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract(target, ".")
    print("Done.")


def _dl_progress(count, block_size, total_size):
    pct = count * block_size * 100 / total_size
    print(f"\r  {pct:.1f}%", end="", flush=True)


# ── Preprocessor (Section III-B) ───────────────────────────────────────
class TextPreprocessor:
    EMOJI_MAP = {
        r":\)|:-\)": " smile ",
        r":\(|:-\(": " sad ",
        r":D|:-D": " laugh ",
        r":P|:-P": " tongue ",
        r"<3": " love ",
        r":O|:-O": " surprise ",
    }

    def clean(self, text: str) -> str:
        if not isinstance(text, str) or pd.isna(text):
            return ""
        text = str(text).strip().lower()
        text = re.sub(r"http\S+|www\S+|https\S+", " [URL] ", text)
        text = re.sub(r"@\w+", " [USER] ", text)
        for pat, rep in self.EMOJI_MAP.items():
            text = re.sub(pat, rep, text)
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)
        text = re.sub(r"[^\w\s.,!?;:'\"-]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


# ── Dataset handler ────────────────────────────────────────────────────
class CyberbullyingDataset:
    """Load and merge CSV files from a directory or single file."""

    def __init__(self, data_path: str, config: Dict = None):
        self.data_path = data_path
        self.config = config or {}
        self.data_cfg = self.config.get("data", {}) if isinstance(self.config, dict) else {}
        self.preprocessor = TextPreprocessor()

    def _has_csv_files(self) -> bool:
        if os.path.isfile(self.data_path) and self.data_path.endswith(".csv"):
            return True
        if os.path.isdir(self.data_path):
            return any(f.endswith(".csv") for f in os.listdir(self.data_path))
        return False

    def _try_download_from_url(self) -> bool:
        url = str(self.data_cfg.get("shahane_dataset_url", "")).strip()
        if not url:
            return False
        try:
            filename = os.path.basename(url.split("?")[0]) or "shahane_dataset.zip"
            target = os.path.join(self.data_path, filename)
            print(f"Attempting Shahane dataset download from URL: {url}")
            urllib.request.urlretrieve(url, target, _dl_progress)
            print()

            if target.lower().endswith(".zip"):
                print(f"Extracting {target} ...")
                with zipfile.ZipFile(target, "r") as zf:
                    zf.extractall(self.data_path)
            return self._has_csv_files()
        except Exception as e:
            print(f"URL-based dataset download failed: {e}")
            return False

    def _try_download_from_kaggle(self) -> bool:
        dataset_id = str(
            self.data_cfg.get("shahane_kaggle_dataset", "saurabhshahane/cyberbullying-dataset")
        ).strip()
        cmd = [
            "kaggle", "datasets", "download", "-d", dataset_id,
            "-p", self.data_path, "--unzip",
        ]
        try:
            print(f"Attempting Shahane dataset download via Kaggle: {dataset_id}")
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if proc.returncode != 0:
                stderr = (proc.stderr or "").strip()
                stdout = (proc.stdout or "").strip()
                msg = stderr if stderr else stdout
                print(f"Kaggle download failed: {msg}")
                return False
            return self._has_csv_files()
        except FileNotFoundError:
            print("Kaggle CLI not found. Install it with: pip install kaggle")
            return False
        except Exception as e:
            print(f"Kaggle download failed: {e}")
            return False

    def _ensure_dataset_available(self):
        if self._has_csv_files():
            return

        auto_download = bool(self.data_cfg.get("auto_download_shahane", True))
        path_basename = os.path.basename(os.path.normpath(self.data_path)).lower()
        if not auto_download or path_basename != "dataset":
            return

        os.makedirs(self.data_path, exist_ok=True)
        print(
            f"No CSV dataset found in {self.data_path}. "
            "Trying automatic Shahane dataset download ..."
        )

        if self._try_download_from_url():
            print("Shahane dataset download succeeded (URL source).")
            return
        if self._try_download_from_kaggle():
            print("Shahane dataset download succeeded (Kaggle source).")
            return

        print(
            "Automatic Shahane download failed. "
            "Place dataset CSV files in dataset/ or set data.shahane_dataset_url in config."
        )

    def load_and_combine(self) -> pd.DataFrame:
        self._ensure_dataset_available()
        frames: List[pd.DataFrame] = []

        paths: List[str] = []
        if os.path.isfile(self.data_path):
            paths.append(self.data_path)
        elif os.path.isdir(self.data_path):
            for f in sorted(os.listdir(self.data_path)):
                if f.endswith(".csv"):
                    paths.append(os.path.join(self.data_path, f))

        for p in paths:
            try:
                df = pd.read_csv(p)
                text_col, label_col = self._detect_columns(df)
                if text_col and label_col:
                    df = df.rename(columns={text_col: "text", label_col: "label"})
                    frames.append(df[["text", "label"]])
                    print(f"  Loaded {p}: {len(df)} rows  text={text_col}  label={label_col}")
            except Exception as e:
                print(f"  Skipping {p}: {e}")

        if not frames:
            raise FileNotFoundError(
                f"No CSV files with valid text/label columns found in {self.data_path}. "
                "Place your dataset CSVs in this directory and re-run."
            )

        combined = pd.concat(frames, ignore_index=True)
        combined["text"] = combined["text"].apply(self.preprocessor.clean)
        combined = combined[combined["text"].str.strip() != ""]

        before_dedup = len(combined)
        combined = combined.drop_duplicates(subset=["text"])
        after_dedup = len(combined)
        if before_dedup != after_dedup:
            print(f"  Deduplication: {before_dedup} → {after_dedup} "
                  f"(removed {before_dedup - after_dedup} duplicates)")

        combined["label"] = combined["label"].apply(self._to_binary)
        combined = combined.dropna(subset=["label"])
        combined["label"] = combined["label"].astype(int)

        pos = int(combined["label"].sum())
        neg = len(combined) - pos
        print(f"Dataset: {len(combined)} samples  ({pos} positive, {neg} negative)")
        return combined

    @staticmethod
    def _detect_columns(df: pd.DataFrame):
        cols_lower = {c.lower(): c for c in df.columns}
        text_col = label_col = None

        for key in ("text", "tweet_text", "tweet", "content", "comment_text"):
            if key in cols_lower:
                text_col = cols_lower[key]
                break
        if text_col is None:
            for c in df.columns:
                if "text" in c.lower() or "tweet" in c.lower() or "content" in c.lower():
                    text_col = c
                    break

        if "oh_label" in cols_lower:
            label_col = cols_lower["oh_label"]
        else:
            for key in ("label", "class", "target", "cyberbullying_type"):
                if key in cols_lower:
                    label_col = cols_lower[key]
                    break
            if label_col is None:
                for c in df.columns:
                    cl = c.lower()
                    if ("label" in cl or "class" in cl or "target" in cl) and not cl.startswith("ed_label"):
                        label_col = c
                        break

        return text_col, label_col

    @staticmethod
    def _to_binary(val) -> Optional[int]:
        if pd.isna(val):
            return None
        s = str(val).lower().strip()
        if s in ("1", "yes", "true", "positive", "bullying", "toxic", "hate"):
            return 1
        if s in ("0", "no", "false", "negative", "non-bullying", "normal", "clean",
                  "not_cyberbullying"):
            return 0
        try:
            return 1 if float(s) > 0.5 else 0
        except ValueError:
            return None

    def create_leakage_proof_splits(self, test_size=0.15, val_size=0.15):
        data = self.load_and_combine()
        texts = data["text"].tolist()
        labels = data["label"].tolist()

        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels, test_size=test_size, stratify=labels, random_state=42,
        )
        val_frac = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_frac, stratify=y_temp, random_state=42,
        )
        print(f"Splits — train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    @staticmethod
    def _synthetic() -> pd.DataFrame:
        rows = [
            ("You are so stupid and worthless!", 1),
            ("I hate you, go die!", 1),
            ("This is a nice day", 0),
            ("Great job on the project", 0),
            ("Nobody likes you", 1),
            ("Have a wonderful day", 0),
        ]
        expanded = [(f"{t} {i}", l) for i in range(100) for t, l in rows]
        return pd.DataFrame(expanded, columns=["text", "label"])


# ── PyTorch dataset ────────────────────────────────────────────────────
class CyberbullyingTorchDataset(Dataset):
    SPECIAL = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}

    def __init__(self, texts: List[str], labels: List[int],
                 vocab: Dict[str, int] = None,
                 max_len: int = 128, max_vocab: int = 20000):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.vocab = vocab if vocab is not None else self._build_vocab(texts, max_vocab)
        for tok, idx in self.SPECIAL.items():
            self.vocab.setdefault(tok, idx)

    def _build_vocab(self, texts: List[str], max_vocab: int) -> Dict[str, int]:
        counter = Counter()
        for t in texts:
            counter.update(self._tokenize(t))
        vocab = dict(self.SPECIAL)
        for word, _ in counter.most_common(max_vocab):
            if word not in vocab:
                vocab[word] = len(vocab)
        return vocab

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        try:
            return word_tokenize(text.lower())
        except Exception:
            return text.lower().split()

    def _encode(self, text: str) -> List[int]:
        tokens = self._tokenize(text)
        tokens = ["<BOS>"] + tokens[: self.max_len - 2] + ["<EOS>"]
        ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]
        pad = self.max_len - len(ids)
        return ids + [self.vocab["<PAD>"]] * pad if pad > 0 else ids[: self.max_len]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        seq = self._encode(self.texts[idx])
        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
            self.texts[idx],
        )

    def get_vocab_size(self) -> int:
        return len(self.vocab)
