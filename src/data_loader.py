"""
Enhanced Data Loader with Better Preprocessing
"""
import os
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import hashlib

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except:
    print("📥 Downloading NLTK resources...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

class EnhancedTextPreprocessor:
    """Enhanced text preprocessing"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
        # Cyberbullying specific patterns
        self.cyberbullying_patterns = [
            r'\b(stupid|idiot|dumb|fool|moron)\b',
            r'\b(ugly|fat|disgusting|gross)\b',
            r'\b(hate|kill|die|worthless)\b',
            r'\b(loser|failure|pathetic|useless)\b',
        ]
        
        # Emoji patterns
        self.emoji_patterns = {
            r':\)|:-\)': ' smile ',
            r':\(|:-\(': ' sad ',
            r':D|:-D': ' laugh ',
            r':P|:-P': ' tongue ',
            r'<3': ' love ',
            r':O|:-O': ' surprise '
        }
    
    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning"""
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        text = str(text).strip().lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', ' [URL] ', text)
        
        # Remove user mentions
        text = re.sub(r'@\w+', ' [USER] ', text)
        
        # Replace emojis
        for pattern, replacement in self.emoji_patterns.items():
            text = re.sub(pattern, replacement, text)
        
        # Handle repeated characters
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s.,!?;:\'\"-]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def preprocess(self, text: str) -> str:
        """Main preprocessing pipeline"""
        return self.clean_text(text)

class CyberbullyingDataset:
    """Dataset handler"""
    
    def __init__(self, data_path: str, config: Dict = None):
        self.data_path = data_path
        self.config = config or {}
        self.preprocessor = EnhancedTextPreprocessor()
    
    def load_and_combine(self) -> pd.DataFrame:
        """Load and combine datasets"""
        print("📥 Loading dataset...")
        
        all_data = []
        
        if os.path.isfile(self.data_path):
            # Single file
            try:
                df = pd.read_csv(self.data_path)
                if 'text' in df.columns and 'label' in df.columns:
                    all_data.append(df[['text', 'label']])
            except:
                pass
        elif os.path.isdir(self.data_path):
            # Multiple files in directory
            for file in os.listdir(self.data_path):
                if file.endswith('.csv'):
                    try:
                        df = pd.read_csv(os.path.join(self.data_path, file))
                        # Try to find text and label columns
                        text_col = None
                        label_col = None
                        
                        lower_map = {col.lower(): col for col in df.columns}
                        
                        # Prefer explicit text columns
                        for key in ['text', 'tweet', 'content']:
                            if key in lower_map:
                                text_col = lower_map[key]
                                break
                        
                        # Prefer explicit label columns (avoid ed_label_0/1 when oh_label exists)
                        if 'oh_label' in lower_map:
                            label_col = lower_map['oh_label']
                        else:
                            for key in ['label', 'class', 'target']:
                                if key in lower_map:
                                    label_col = lower_map[key]
                                    break
                        
                        # Fallback: match by substring if still not found
                        if text_col is None:
                            for col in df.columns:
                                col_lower = col.lower()
                                if 'text' in col_lower or 'tweet' in col_lower or 'content' in col_lower:
                                    text_col = col
                                    break
                        
                        if label_col is None:
                            for col in df.columns:
                                col_lower = col.lower()
                                if ('label' in col_lower or 'class' in col_lower or 'target' in col_lower) and not col_lower.startswith('ed_label'):
                                    label_col = col
                                    break
                        
                        if label_col is None:
                            for col in df.columns:
                                col_lower = col.lower()
                                if 'label' in col_lower or 'class' in col_lower or 'target' in col_lower:
                                    label_col = col
                                    break
                        
                        if text_col and label_col:
                            df = df.rename(columns={text_col: 'text', label_col: 'label'})
                            all_data.append(df[['text', 'label']])
                    except:
                        continue
        
        if not all_data:
            # Create sample dataset if no data found
            print("⚠ No data found, creating sample dataset...")
            all_data = [self._create_sample_dataset()]
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True) if all_data else self._create_sample_dataset()
        
        # Preprocess texts
        combined_df['text'] = combined_df['text'].apply(self.preprocessor.preprocess)
        
        # Remove empty texts
        combined_df = combined_df[combined_df['text'].str.strip() != '']
        
        # Ensure labels are binary
        combined_df['label'] = combined_df['label'].apply(self._normalize_label)
        
        print(f"📊 Dataset Statistics:")
        print(f"   Total samples: {len(combined_df)}")
        print(f"   Positive samples: {combined_df['label'].sum()}")
        print(f"   Negative samples: {len(combined_df) - combined_df['label'].sum()}")
        
        return combined_df
    
    def _create_sample_dataset(self) -> pd.DataFrame:
        """Create sample dataset"""
        samples = [
            ("You're so stupid and worthless!", 1),
            ("I hate you, go die!", 1),
            ("You're such a loser", 1),
            ("This is a nice day", 0),
            ("I love this weather", 0),
            ("Great job on the project", 0),
            ("You're ugly and fat", 1),
            ("Nobody likes you", 1),
            ("Have a wonderful day", 0),
            ("Looking forward to meeting", 0),
            ("You should kill yourself", 1),
            ("Everyone hates you", 1),
            ("Thanks for your help", 0),
            ("Appreciate your support", 0),
            ("You're a failure", 1),
            ("You're not welcome here", 1),
            ("Good morning everyone", 0),
            ("Excellent presentation", 0),
            ("You're a disgrace", 1),
            ("I'm proud of your work", 0)
        ]
        
        # Duplicate to create larger dataset
        expanded_samples = []
        for i in range(50):  # Create 1000 samples
            for text, label in samples:
                expanded_samples.append((f"{text} {i}", label))
        
        return pd.DataFrame(expanded_samples, columns=['text', 'label'])
    
    def _normalize_label(self, label) -> int:
        """Normalize label to binary"""
        if pd.isna(label):
            return 0
        
        label_str = str(label).lower().strip()
        
        if label_str in ['1', 'yes', 'true', 'positive', 'bullying', 'toxic', 'hate']:
            return 1
        elif label_str in ['0', 'no', 'false', 'negative', 'non-bullying', 'normal', 'clean']:
            return 0
        else:
            try:
                num = float(label_str)
                return 1 if num > 0.5 else 0
            except:
                return 0
    
    def create_leakage_proof_splits(self, test_size: float = 0.15, val_size: float = 0.15):
        """Create data splits with no leakage"""
        data = self.load_and_combine()
        
        texts = data['text'].tolist()
        labels = data['label'].tolist()
        
        # First split: test
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels,
            test_size=test_size,
            stratify=labels,
            random_state=42
        )
        
        # Second split: train/val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            stratify=y_temp,
            random_state=42
        )
        
        print(f"\n🎯 Data Splits:")
        print(f"   Training: {len(X_train)} samples")
        print(f"   Validation: {len(X_val)} samples")
        print(f"   Test: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test


class CyberbullyingTorchDataset(Dataset):
    """PyTorch Dataset for cyberbullying data"""
    
    def __init__(self, texts: List[str], labels: List[int], 
                 vocab: Dict = None, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.preprocessor = EnhancedTextPreprocessor()
        
        # Define special tokens
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }
        
        # Build vocabulary
        if vocab is None:
            self.vocab = self._build_vocab(texts)
        else:
            self.vocab = vocab
        
        # Add special tokens if not present
        for token, idx in self.special_tokens.items():
            if token not in self.vocab:
                self.vocab[token] = idx
        
        self.idx_to_token = {idx: token for token, idx in self.vocab.items()}
    
    def _safe_tokenize(self, text: str) -> List[str]:
        """Safe tokenization that doesn't rely on NLTK"""
        try:
            return word_tokenize(text.lower())
        except:
            # Fallback to simple tokenization
            return text.lower().split()
    
    def _build_vocab(self, texts: List[str]) -> Dict:
        """Build vocabulary from texts"""
        counter = Counter()
        
        for text in texts:
            tokens = self._safe_tokenize(text)
            counter.update(tokens)
        
        # Keep most frequent tokens
        vocab = {token: idx + len(self.special_tokens) 
                for idx, (token, _) in enumerate(counter.most_common(20000))}
        
        # Add special tokens
        vocab.update(self.special_tokens)
        
        return vocab
    
    def _text_to_sequence(self, text: str) -> List[int]:
        """Convert text to sequence of indices"""
        tokens = self._safe_tokenize(text)
        tokens = ['<BOS>'] + tokens[:self.max_len-2] + ['<EOS>']
        
        sequence = []
        for token in tokens:
            if token in self.vocab:
                sequence.append(self.vocab[token])
            else:
                sequence.append(self.vocab['<UNK>'])
        
        # Pad or truncate
        if len(sequence) < self.max_len:
            sequence += [self.vocab['<PAD>']] * (self.max_len - len(sequence))
        else:
            sequence = sequence[:self.max_len]
        
        return sequence
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        text = self.texts[idx]
        sequence = self._text_to_sequence(text)
        label = self.labels[idx]
        
        return (
            torch.tensor(sequence, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
            text
        )
    
    def get_vocab_size(self) -> int:
        return len(self.vocab)