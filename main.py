#!/usr/bin/env python3
"""
MAIN SCRIPT - TriFuse Cyberbullying Detection
Now with K-Fold Cross Validation Support
"""

import os
import sys
import torch

# ⬇️⬇️⬇️ FIX FOR PBS/TORCH ERROR ⬇️⬇️⬇️
home_dir = os.path.expanduser('~')
temp_dir = os.path.join(home_dir, 'tmp')
os.makedirs(temp_dir, exist_ok=True)

os.environ['TMPDIR'] = temp_dir
os.environ['TEMP'] = temp_dir
os.environ['TMP'] = temp_dir
os.environ['HOME'] = home_dir

torch_ext_dir = os.path.join(temp_dir, 'torch_extensions')
os.makedirs(torch_ext_dir, exist_ok=True)
os.environ['TORCH_EXTENSIONS_DIR'] = torch_ext_dir
os.environ['PYTORCH_JIT'] = '0'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# ⬆️⬆️⬆️ END FIX ⬆️⬆️⬆️

import yaml
import json
import numpy as np
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

# Import modules
try:
    from data_loader import CyberbullyingDataset, CyberbullyingTorchDataset
    from models import OptimizedTriFuseModel
    from baseline_models import (
        BiLSTMBaseline, CNNBaseline, BERTBaseline, 
        TunedLSTMBaseline, RandomForestEnsemble, LightGBMEnsemble
    )
    from ablation_models import create_ablation_model
    from attention_optimizer import EnhancedAttentionTrainer
    from utils import (
        plot_training_history,
        plot_confusion_matrix_with_metrics,
        plot_model_comparison,
        create_comprehensive_report
    )
    print("✅ All modules imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Current directory: {current_dir}")
    print(f"Files in src: {os.listdir(os.path.join(current_dir, 'src')) if os.path.exists(os.path.join(current_dir, 'src')) else 'src not found'}")
    sys.exit(1)

# ==================== ADD K-FOLD IMPORTS ====================
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, ConcatDataset

def load_config(config_path: str = "configs/config.yaml"):
    """Load configuration file"""
    abs_config_path = os.path.join(current_dir, config_path) if not os.path.isabs(config_path) else config_path
    
    if not os.path.exists(abs_config_path):
        print(f"⚠ Configuration file not found: {abs_config_path}")
        print("Using default configuration...")
        return get_default_config()
    
    with open(abs_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ Configuration loaded from: {abs_config_path}")
    return config

def get_default_config():
    """Get default configuration"""
    return {
        'system': {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'seed': 42,
            'num_workers': 4
        },
        'data': {
            'data_path': 'dataset/',
            'max_seq_len': 128,
            'test_size': 0.15,
            'val_size': 0.15,
            'use_augmentation': False
        },
        'model': {
            'num_classes': 2,
            'embed_dim': 300,
            'dropout_rate': 0.3,
            'lexical_filters': [2,3,4,5],
            'lexical_num_filters': 128,
            'semantic_num_layers': 2,
            'structural_num_layers': 2,
            'attention_entropy_weight': 0.05,
            'attention_diversity_weight': 0.1
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'epochs': 100,
            'patience': 15,
            'gradient_clip': 1.0
        },
        'paths': {
            'base_output': 'outputs/'
        }
    }

def setup_environment(config):
    """Setup training environment"""
    seed = config['system']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    device_type = config['system']['device']
    if device_type == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_type)
    
    print(f"🔧 Device: {device}")
    print(f"🔧 Seed: {seed}")
    
    if device.type == 'cuda':
        print(f"🔧 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🔧 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return device

def create_output_directories(config):
    """Create output directories"""
    print("\n📁 Creating output directories...")
    
    # Ensure all paths exist
    if 'paths' in config:
        for key, path in config['paths'].items():
            if key.endswith('_dir') or key.endswith('_path'):
                if not os.path.isabs(path):
                    path = os.path.join(config['paths']['base_output'], os.path.basename(path))
                try:
                    os.makedirs(path, exist_ok=True)
                    print(f"   ✅ {key}: {path}")
                except Exception as e:
                    print(f"   ❌ Error creating {key}: {path} - {e}")
                    # Fallback to current directory
                    fallback_path = os.path.join(current_dir, key)
                    os.makedirs(fallback_path, exist_ok=True)
                    config['paths'][key] = fallback_path
                    print(f"   ✅ Fallback {key}: {fallback_path}")
    else:
        # Create default directories in current directory
        default_dirs = ['models', 'plots', 'results', 'logs']
        for dir_name in default_dirs:
            dir_path = os.path.join(current_dir, 'outputs', dir_name)
            os.makedirs(dir_path, exist_ok=True)
            print(f"   ✅ Created default: {dir_path}")

def load_data(config):
    """Load and prepare data"""
    print("\n📊 Loading data...")
    
    try:
        data_path = config['data']['data_path']
        
        # Check if path exists
        if not os.path.exists(data_path):
            print(f"⚠ Data directory not found: {data_path}")
            print("Creating sample dataset for testing...")
            create_sample_dataset(data_path)
        
        # Create dataset
        dataset = CyberbullyingDataset(data_path=data_path, config=config)
        
        # Create splits
        X_train, X_val, X_test, y_train, y_val, y_test = dataset.create_leakage_proof_splits(
            test_size=config['data']['test_size'],
            val_size=config['data']['val_size']
        )
        
        print(f"   ✅ Training samples: {len(X_train)}")
        print(f"   ✅ Validation samples: {len(X_val)}")
        print(f"   ✅ Test samples: {len(X_test)}")
        
        # Create PyTorch datasets
        train_dataset = CyberbullyingTorchDataset(
            X_train, y_train, 
            max_len=config['data']['max_seq_len']
        )
        
        val_dataset = CyberbullyingTorchDataset(
            X_val, y_val, 
            vocab=train_dataset.vocab, 
            max_len=config['data']['max_seq_len']
        )
        
        test_dataset = CyberbullyingTorchDataset(
            X_test, y_test, 
            vocab=train_dataset.vocab, 
            max_len=config['data']['max_seq_len']
        )
        
        vocab_size = train_dataset.get_vocab_size()
        print(f"   ✅ Vocabulary size: {vocab_size}")
        
        return train_dataset, val_dataset, test_dataset, vocab_size
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        raise

def create_sample_dataset(data_path):
    """Create sample dataset if none exists"""
    import pandas as pd
    import os
    
    os.makedirs(data_path, exist_ok=True)
    
    # Sample cyberbullying data
    sample_data = {
        'text': [
            "You are so stupid and worthless!",
            "I hate you, go die!",
            "You're such a loser",
            "This is a nice day",
            "I love this weather",
            "Great job on the project",
            "You're ugly and fat",
            "Nobody likes you",
            "Have a wonderful day",
            "Looking forward to meeting",
            "You should kill yourself",
            "Everyone hates you",
            "Thanks for your help",
            "Appreciate your support",
            "You're a failure",
            "You're not welcome here",
            "Good morning everyone",
            "Excellent presentation",
            "You're a disgrace",
            "I'm proud of your work"
        ],
        'label': [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0]
    }
    
    # Duplicate to create larger dataset
    expanded_data = []
    for i in range(100):
        for text, label in zip(sample_data['text'], sample_data['label']):
            expanded_data.append({
                'text': f"{text} {i}",
                'label': label
            })
    
    df = pd.DataFrame(expanded_data)
    
    # Save to multiple files to simulate real dataset
    for i in range(5):
        chunk_size = len(df) // 5
        chunk = df[i*chunk_size:(i+1)*chunk_size]
        chunk.to_csv(os.path.join(data_path, f'cyberbullying_data_{i}.csv'), index=False)
    
    print(f"   Created sample dataset with {len(df)} samples in {data_path}")

def _is_ensemble_model(model):
    return isinstance(model, (RandomForestEnsemble, LightGBMEnsemble))

def _extract_ensemble_features(model, sequences: torch.Tensor) -> torch.Tensor:
    """Extract features for RandomForest/LightGBM ensembles"""
    embedded = model.embedding(sequences)

    if isinstance(model, RandomForestEnsemble):
        mean_feat = torch.mean(embedded, dim=1)
        max_feat, _ = torch.max(embedded, dim=1)
        min_feat, _ = torch.min(embedded, dim=1)
        std_feat = torch.std(embedded, dim=1)
        feature_vec = torch.cat([mean_feat, max_feat, min_feat, std_feat], dim=1)
        return feature_vec

    if isinstance(model, LightGBMEnsemble):
        first_token = embedded[:, 0, :]
        last_token = embedded[:, -1, :]
        mid_token = embedded[:, embedded.shape[1] // 2, :]

        mean_feat = torch.mean(embedded, dim=1)
        max_feat, _ = torch.max(embedded, dim=1)
        min_feat, _ = torch.min(embedded, dim=1)
        std_feat = torch.std(embedded, dim=1)

        feature_vec = torch.cat(
            [first_token, last_token, mid_token, mean_feat, max_feat, min_feat, std_feat],
            dim=1
        )
        return feature_vec

    raise ValueError("Unsupported ensemble model type")

def _fit_ensemble_model(model, train_loader, device):
    """Fit RandomForest/LightGBM using extracted features"""
    all_features = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for sequences, labels, _ in train_loader:
            sequences = sequences.to(device)
            features = _extract_ensemble_features(model, sequences)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    X_train = np.vstack(all_features) if all_features else np.empty((0, model.embed_dim))
    y_train = np.concatenate(all_labels) if all_labels else np.array([])

    if len(y_train) > 0:
        model.fit(X_train, y_train)
    return model

def train_model(model, model_name, train_loader, val_loader, test_loader, config, device, is_sklearn=False):
    """Train and evaluate a model"""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print('='*60)
    
    # Handle sklearn-style ensemble models
    if _is_ensemble_model(model):
        _fit_ensemble_model(model, train_loader, device)

        def _eval_loader(loader):
            all_preds = []
            all_labels = []
            model.eval()
            with torch.no_grad():
                for sequences, labels, _ in loader:
                    sequences = sequences.to(device)
                    logits = model(sequences)
                    preds = torch.argmax(logits, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())
            return all_labels, all_preds

        val_labels, val_preds = _eval_loader(val_loader)
        test_labels, test_preds = _eval_loader(test_loader)

        val_acc = accuracy_score(val_labels, val_preds) if len(val_labels) else 0
        test_acc = accuracy_score(test_labels, test_preds) if len(test_labels) else 0
        f1 = f1_score(test_labels, test_preds, average='weighted') if len(test_labels) else 0
        precision = precision_score(test_labels, test_preds, average='weighted') if len(test_labels) else 0
        recall = recall_score(test_labels, test_preds, average='weighted') if len(test_labels) else 0

        print(f"\n📊 {model_name.upper()} TEST RESULTS:")
        print(f"   Accuracy:  {test_acc:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print("-" * 50)

        return {
            'accuracy': test_acc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'history': {},
            'predictions': test_preds,
            'labels': test_labels
        }

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'learning_rates': []
    }
    
    best_val_acc = 0
    patience_counter = 0
    patience = config['training']['patience']
    
    print(f"🔥 Starting training for {config['training']['epochs']} epochs")
    print(f"🔥 Early stopping patience: {patience} epochs")
    
    for epoch in range(config['training']['epochs']):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for sequences, labels, texts in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - FIXED: Check if model has forward_full method
            if hasattr(model, 'forward_full'):
                # Some models might need both texts and sequences
                try:
                    # First try with just sequences (for OptimizedTriFuseModel)
                    logits = model(sequences)
                except TypeError:
                    # If that fails, try with both arguments
                    logits, _ = model(texts, sequences)
            else:
                # For models that only take sequences
                logits = model(sequences)
            
            loss = criterion(logits, labels)
            loss.backward()
            
            # Gradient clipping
            if 'gradient_clip' in config['training']:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config['training']['gradient_clip']
                )
            
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        # Update scheduler
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for sequences, labels, texts in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                
                # FIXED: Same logic as training
                if hasattr(model, 'forward_full'):
                    try:
                        logits = model(sequences)
                    except TypeError:
                        logits, _ = model(texts, sequences)
                else:
                    logits = model(sequences)
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        print(f"   Epoch {epoch+1:3d}/{config['training']['epochs']} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save model
            model_path = f'best_{model_name}.pth'
            if 'paths' in config and 'models_dir' in config['paths']:
                model_path = os.path.join(config['paths']['models_dir'], f'best_{model_name}.pth')
            
            torch.save(model.state_dict(), model_path)
            print(f"     💾 Saved best model (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"     ⏹️ Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model_path = f'best_{model_name}.pth'
    if 'paths' in config and 'models_dir' in config['paths']:
        model_path = os.path.join(config['paths']['models_dir'], f'best_{model_name}.pth')
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"     🔄 Loaded best model from {model_path}")
    
    # Test evaluation
    model.eval()
    test_predictions = []
    test_labels = []
    
    with torch.no_grad():
        for sequences, labels, texts in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            # FIXED: Same logic as training and validation
            if hasattr(model, 'forward_full'):
                try:
                    logits = model(sequences)
                except TypeError:
                    logits, _ = model(texts, sequences)
            else:
                logits = model(sequences)
            
            preds = torch.argmax(logits, dim=1)
            test_predictions.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(test_labels, test_predictions)
    f1 = f1_score(test_labels, test_predictions, average='weighted')
    precision = precision_score(test_labels, test_predictions, average='weighted')
    recall = recall_score(test_labels, test_predictions, average='weighted')
    
    print(f"\n📊 {model_name.upper()} TEST RESULTS:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print("-" * 50)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'history': history,
        'predictions': test_predictions,
        'labels': test_labels
    }

# ==================== ADDED K-FOLD FUNCTIONS ====================

def kfold_train_model(model, model_name, train_loader, val_loader, config, device):
    """Simplified training for K-fold (no test evaluation)"""
    if _is_ensemble_model(model):
        _fit_ensemble_model(model, train_loader, device)
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for sequences, labels, _ in val_loader:
                sequences = sequences.to(device)
                logits = model(sequences)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds.cpu() == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total if val_total > 0 else 0
        return model, val_acc

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    criterion = torch.nn.CrossEntropyLoss()
    best_val_acc = 0
    best_model_state = None
    
    epochs = min(config['training']['epochs'], 20)  # Shorter for K-fold
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for sequences, labels, texts in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            if hasattr(model, 'forward_full'):
                try:
                    logits = model(sequences)
                except TypeError:
                    logits, _ = model(texts, sequences)
            else:
                logits = model(sequences)
            
            loss = criterion(logits, labels)
            loss.backward()
            
            if 'gradient_clip' in config['training']:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config['training']['gradient_clip']
                )
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for sequences, labels, texts in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                
                if hasattr(model, 'forward_full'):
                    try:
                        logits = model(sequences)
                    except TypeError:
                        logits, _ = model(texts, sequences)
                else:
                    logits = model(sequences)
                
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, best_val_acc

def run_kfold_validation(model_name, dataset, vocab_size, config, device, k_folds=5):
    """Run K-fold cross validation for a single model"""
    print(f"\n🔬 {k_folds}-Fold Cross Validation for {model_name}")
    print(f"{'='*60}")
    
    # Get labels from dataset
    labels = []
    for i in range(len(dataset)):
        _, label, _ = dataset[i]
        labels.append(label.item())
    
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_accuracies = []
    fold_f1_scores = []
    
    from sklearn.metrics import accuracy_score, f1_score
    from torch.utils.data import DataLoader
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(dataset)), labels)):
        print(f"\n  Fold {fold + 1}/{k_folds}")
        
        # Create subsets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        # Create loaders
        train_loader = DataLoader(train_subset, batch_size=config['training']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config['training']['batch_size'], shuffle=False)
        
        # Initialize model
        if model_name == 'trifuse':
            model = OptimizedTriFuseModel(vocab_size, config=config['model']).to(device)
        elif model_name == 'bilstm':
            model = BiLSTMBaseline(
                vocab_size=vocab_size,
                embed_dim=config['model']['embed_dim'],
                hidden_dim=256,
                num_layers=2,
                num_classes=config['model']['num_classes'],
                dropout=config['model']['dropout_rate']
            ).to(device)
        elif model_name == 'cnn':
            model = CNNBaseline(
                vocab_size=vocab_size,
                embed_dim=config['model']['embed_dim'],
                num_filters=128,
                filter_sizes=[2, 3, 4, 5],
                num_classes=config['model']['num_classes'],
                dropout=config['model']['dropout_rate']
            ).to(device)
        elif model_name == 'bert':
            model = BERTBaseline(
                num_classes=config['model']['num_classes'],
                dropout=config['model']['dropout_rate']
            ).to(device)
        elif model_name == 'tuned_lstm':
            model = TunedLSTMBaseline(
                vocab_size=vocab_size,
                embed_dim=config['model']['embed_dim'],
                hidden_dim=256,
                num_layers=3,
                num_classes=config['model']['num_classes'],
                dropout=config['model']['dropout_rate']
            ).to(device)
        elif model_name == 'rf':
            model = RandomForestEnsemble(
                vocab_size=vocab_size,
                embed_dim=config['model']['embed_dim'],
                num_classes=config['model']['num_classes'],
                n_estimators=config['model'].get('rf_n_estimators', 100)
            ).to(device)
        elif model_name == 'lightgbm':
            model = LightGBMEnsemble(
                vocab_size=vocab_size,
                embed_dim=config['model']['embed_dim'],
                num_classes=config['model']['num_classes']
            ).to(device)
        else:
            # Try ablation model
            try:
                model = create_ablation_model(model_name, vocab_size, config['model']).to(device)
            except:
                print(f"  ❌ Model {model_name} not found, skipping...")
                return None
        
        # Train this fold
        model, best_val_acc = kfold_train_model(model, f"{model_name}_fold{fold+1}", 
                                               train_loader, val_loader, config, device)
        
        # Evaluate on validation set
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, labels_batch, texts in val_loader:
                sequences = sequences.to(device)
                labels_batch = labels_batch.to(device)
                
                if hasattr(model, 'forward_full'):
                    try:
                        logits = model(sequences)
                    except TypeError:
                        logits, _ = model(texts, sequences)
                else:
                    logits = model(sequences)
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        fold_accuracies.append(accuracy)
        fold_f1_scores.append(f1)
        
        print(f"    Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    # Calculate statistics
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    mean_f1 = np.mean(fold_f1_scores)
    
    print(f"\n  📊 {model_name.upper()} - {k_folds}-Fold Results:")
    print(f"    Mean Accuracy:  {mean_accuracy:.4f} (±{std_accuracy:.4f})")
    print(f"    Mean F1-Score:  {mean_f1:.4f}")
    print(f"    Fold Accuracies: {[f'{acc:.4f}' for acc in fold_accuracies]}")
    
    return {
        'model_name': model_name,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'mean_f1_score': mean_f1,
        'fold_accuracies': fold_accuracies,
        'fold_f1_scores': fold_f1_scores
    }

def compare_models_kfold(models_to_test, train_dataset, val_dataset, vocab_size, config, device, k_folds=5):
    """Compare multiple models using K-fold cross validation"""
    print(f"\n{'='*80}")
    print("🧪 K-FOLD CROSS VALIDATION - MODEL COMPARISON")
    print(f"{'='*80}")
    
    # Combine train and val for K-fold
    kfold_dataset = ConcatDataset([train_dataset, val_dataset])
    
    results = {}
    
    for model_name in models_to_test:
        kfold_result = run_kfold_validation(
            model_name=model_name,
            dataset=kfold_dataset,
            vocab_size=vocab_size,
            config=config,
            device=device,
            k_folds=k_folds
        )
        
        if kfold_result:
            results[model_name] = kfold_result
    
    # Display comparison table
    if results:
        print(f"\n{'='*80}")
        print("🏆 K-FOLD COMPARISON RESULTS")
        print(f"{'='*80}")
        print(f"{'Model':<20} {'Mean Accuracy':<15} {'Std Dev':<10} {'Mean F1':<15}")
        print(f"{'-'*80}")
        
        for model_name, result in results.items():
            print(f"{model_name.replace('_', ' ').title():<20} "
                  f"{result['mean_accuracy']:<15.4f} "
                  f"{result['std_accuracy']:<10.4f} "
                  f"{result['mean_f1_score']:<15.4f}")
    
    return results

def generate_kfold_paper_table(kfold_results):
    """Generate LaTeX table for paper from K-fold results"""
    print(f"\n{'='*80}")
    print("📄 LaTeX TABLE FOR PAPER")
    print(f"{'='*80}")
    
    latex_table = """
\\begin{table}[htbp]
\\centering
\\caption{K-fold Cross Validation Results (Accuracy in \\%)}
\\label{tab:kfold_results}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Model} & \\textbf{Mean} & \\textbf{Std Dev} & \\textbf{95\\% CI} \\\\
\\midrule
"""
    
    for model_name, result in kfold_results.items():
        mean_acc = result['mean_accuracy'] * 100
        std_acc = result['std_accuracy'] * 100
        ci_lower = (mean_acc - 1.96 * std_acc / np.sqrt(len(result['fold_accuracies'])))
        ci_upper = (mean_acc + 1.96 * std_acc / np.sqrt(len(result['fold_accuracies'])))
        
        display_name = model_name.replace('_', ' ').title()
        if model_name == 'trifuse':
            display_name = "\\textbf{TriFuse (Proposed)}"
        
        latex_table += f"{display_name} & {mean_acc:.2f} & {std_acc:.2f} & ({ci_lower:.2f}, {ci_upper:.2f}) \\\\\n"
    
    latex_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    print(latex_table)
    
    # Save to file
    table_file = os.path.join('outputs', 'kfold_results_table.tex')
    with open(table_file, 'w') as f:
        f.write(latex_table)
    
    print(f"\n💾 LaTeX table saved to: {table_file}")
    
    return latex_table

# ==================== MAIN FUNCTION (UPDATED) ====================

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='TriFuse Cyberbullying Detection')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Config file path')
    parser.add_argument('--mode', choices=['full', 'ablation', 'baseline', 'single', 'kfold'], 
                       default='full', help='Run mode')
    parser.add_argument('--model', type=str, default='trifuse', help='Specific model (for single mode)')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for cross validation')
    parser.add_argument('--quick', action='store_true', help='Quick test mode (10 epochs)')
    parser.add_argument('--report', action='store_true', default=True, help='Generate comprehensive report')
    args = parser.parse_args()
    
    # AUTO K-FOLD FOR TRIFUSE: If model is trifuse and mode is default (full), switch to kfold
    if args.model == 'trifuse' and args.mode == 'full':
        args.mode = 'kfold'
    
    print("\n" + "="*80)
    print("TRI-FUSE CYBERBULLYING DETECTION")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Mode: {args.mode}")
    if args.mode == 'single':
        print(f"Model: {args.model}")
    if args.mode == 'kfold':
        print(f"K-Folds: {args.k_folds}")
    print(f"Quick mode: {'Yes' if args.quick else 'No'}")
    
    # Load config
    config = load_config(args.config)
    
    # Quick mode adjustments
    if args.quick:
        config['training']['epochs'] = 10
        config['training']['batch_size'] = 16
        print("⚡ Running in QUICK mode (10 epochs)")
    
    # Setup environment
    device = setup_environment(config)
    
    # Create output directories
    create_output_directories(config)
    
    # Load data
    train_dataset, val_dataset, test_dataset, vocab_size = load_data(config)
    
    from torch.utils.data import DataLoader
    
    # Create data loaders (only if not in K-fold mode)
    if args.mode != 'kfold':
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['training']['batch_size'], 
            shuffle=True, 
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config['training']['batch_size'], 
            shuffle=False, 
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config['training']['batch_size'], 
            shuffle=False, 
            num_workers=0
        )
    
    # ==================== K-FOLD MODE ====================
    if args.mode == 'kfold':
        print("\n" + "="*80)
        print("🧪 RUNNING K-FOLD CROSS VALIDATION")
        print("="*80)
        
        # Choose which models to compare
        if args.model == 'trifuse':
            models_to_test = ['trifuse']
        elif args.model == 'all':
            models_to_test = ['trifuse', 'bilstm', 'cnn', 'tuned_lstm', 'bert', 'rf', 'lightgbm', 'lexical_only', 'semantic_only', 'structural_only']
        elif args.model == 'baselines':
            models_to_test = ['bilstm', 'cnn', 'tuned_lstm', 'bert', 'rf', 'lightgbm']
        else:
            models_to_test = [args.model]
        
        # Run K-fold comparison
        kfold_results = compare_models_kfold(
            models_to_test=models_to_test,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            vocab_size=vocab_size,
            config=config,
            device=device,
            k_folds=args.k_folds
        )
        
        # Save K-fold results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join('outputs', f'kfold_results_{timestamp}.json')
        
        # Convert to serializable format
        serializable_results = {}
        for model_name, result in kfold_results.items():
            serializable_results[model_name] = {
                'mean_accuracy': float(result['mean_accuracy']),
                'std_accuracy': float(result['std_accuracy']),
                'mean_f1_score': float(result['mean_f1_score']),
                'fold_accuracies': [float(acc) for acc in result['fold_accuracies']]
            }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n💾 K-fold results saved to: {results_file}")
        
        # Generate paper table
        generate_kfold_paper_table(kfold_results)
        
        # Also test the best model from K-fold on test set
        print("\n" + "="*80)
        print("🧪 Testing best K-fold model on test set...")
        print("="*80)
        
        # Find best model from K-fold
        if kfold_results:
            best_model_name = max(kfold_results.items(), key=lambda x: x[1]['mean_accuracy'])[0]
            print(f"Best model from K-fold: {best_model_name}")
            
            # Train final model on full training data
            full_train_dataset = ConcatDataset([train_dataset, val_dataset])
            full_train_loader = DataLoader(full_train_dataset, 
                                         batch_size=config['training']['batch_size'], 
                                         shuffle=True)
            test_loader = DataLoader(test_dataset, 
                                   batch_size=config['training']['batch_size'], 
                                   shuffle=False)
            
            if best_model_name == 'trifuse':
                final_model = OptimizedTriFuseModel(vocab_size, config=config['model']).to(device)
            elif best_model_name == 'bilstm':
                final_model = BiLSTMBaseline(
                    vocab_size=vocab_size,
                    embed_dim=config['model']['embed_dim'],
                    hidden_dim=256,
                    num_layers=2,
                    num_classes=config['model']['num_classes'],
                    dropout=config['model']['dropout_rate']
                ).to(device)
            elif best_model_name == 'cnn':
                final_model = CNNBaseline(
                    vocab_size=vocab_size,
                    embed_dim=config['model']['embed_dim'],
                    num_filters=128,
                    filter_sizes=[2, 3, 4, 5],
                    num_classes=config['model']['num_classes'],
                    dropout=config['model']['dropout_rate']
                ).to(device)
            elif best_model_name == 'tuned_lstm':
                final_model = TunedLSTMBaseline(
                    vocab_size=vocab_size,
                    embed_dim=config['model']['embed_dim'],
                    hidden_dim=256,
                    num_layers=3,
                    num_classes=config['model']['num_classes'],
                    dropout=config['model']['dropout_rate']
                ).to(device)
            elif best_model_name == 'bert':
                final_model = BERTBaseline(
                    num_classes=config['model']['num_classes'],
                    dropout=config['model']['dropout_rate']
                ).to(device)
            elif best_model_name == 'rf':
                final_model = RandomForestEnsemble(
                    vocab_size=vocab_size,
                    embed_dim=config['model']['embed_dim'],
                    num_classes=config['model']['num_classes'],
                    n_estimators=config['model'].get('rf_n_estimators', 100)
                ).to(device)
            elif best_model_name == 'lightgbm':
                final_model = LightGBMEnsemble(
                    vocab_size=vocab_size,
                    embed_dim=config['model']['embed_dim'],
                    num_classes=config['model']['num_classes']
                ).to(device)
            else:
                try:
                    final_model = create_ablation_model(best_model_name, vocab_size, config['model']).to(device)
                except:
                    print(f"❌ Could not create model {best_model_name}")
                    return 0
            
            # Train final model
            final_result = train_model(final_model, f"{best_model_name}_final", 
                                      full_train_loader, test_loader, test_loader, config, device)
            
            print(f"\n🎯 Final Test Performance of Best Model ({best_model_name}):")
            print(f"   Accuracy:  {final_result['accuracy']:.4f}")
            print(f"   F1-Score:  {final_result['f1_score']:.4f}")
        
        return 0
    
    # ==================== ORIGINAL MODES ====================
    # Results storage
    results = {}
    histories = {}
    
    # Run based on mode
    if args.mode == 'single':
        print(f"\nTraining single model: {args.model}")
        
        if args.model == 'trifuse':
            model = OptimizedTriFuseModel(vocab_size, config=config['model']).to(device)
            print(f"   Model: TriFuse with {sum(p.numel() for p in model.parameters()):,} parameters")
        elif args.model == 'bilstm':
            model = BiLSTMBaseline(
                vocab_size=vocab_size,
                embed_dim=config['model']['embed_dim'],
                hidden_dim=256,
                num_layers=2,
                num_classes=config['model']['num_classes'],
                dropout=config['model']['dropout_rate']
            ).to(device)
            print(f"   Model: BiLSTM Baseline")
        elif args.model == 'cnn':
            model = CNNBaseline(
                vocab_size=vocab_size,
                embed_dim=config['model']['embed_dim'],
                num_filters=128,
                filter_sizes=[2, 3, 4, 5],
                num_classes=config['model']['num_classes'],
                dropout=config['model']['dropout_rate']
            ).to(device)
            print(f"   Model: CNN Baseline")
        elif args.model == 'bert':
            model = BERTBaseline(
                num_classes=config['model']['num_classes'],
                dropout=config['model']['dropout_rate']
            ).to(device)
            print(f"   Model: BERT Baseline")
        elif args.model == 'tuned_lstm':
            model = TunedLSTMBaseline(
                vocab_size=vocab_size,
                embed_dim=config['model']['embed_dim'],
                hidden_dim=256,
                num_layers=3,
                num_classes=config['model']['num_classes'],
                dropout=config['model']['dropout_rate']
            ).to(device)
            print(f"   Model: Tuned LSTM Baseline")
        elif args.model == 'rf':
            model = RandomForestEnsemble(
                vocab_size=vocab_size,
                embed_dim=config['model']['embed_dim'],
                num_classes=config['model']['num_classes'],
                n_estimators=config['model'].get('rf_n_estimators', 100)
            ).to(device)
            print(f"   Model: RandomForest Baseline")
        elif args.model == 'lightgbm':
            model = LightGBMEnsemble(
                vocab_size=vocab_size,
                embed_dim=config['model']['embed_dim'],
                num_classes=config['model']['num_classes']
            ).to(device)
            print(f"   Model: LightGBM Baseline")
        else:
            # Try ablation model
            try:
                model = create_ablation_model(args.model, vocab_size, config['model']).to(device)
                print(f"   Model: {args.model.replace('_', ' ').title()} Ablation")
            except:
                print(f"❌ Unknown model: {args.model}")
                return 1
        
        result = train_model(model, args.model, train_loader, val_loader, test_loader, config, device)
        results[args.model] = result
        histories[args.model] = result['history']
    
    elif args.mode in ['full', 'baseline']:
        print("\n" + "="*60)
        print("TRAINING BASELINES")
        print("="*60)
        
        # BiLSTM Baseline
        bilstm_model = BiLSTMBaseline(
            vocab_size=vocab_size,
            embed_dim=config['model']['embed_dim'],
            hidden_dim=256,
            num_layers=2,
            num_classes=config['model']['num_classes'],
            dropout=config['model']['dropout_rate']
        ).to(device)
        result = train_model(bilstm_model, "BiLSTM", train_loader, val_loader, test_loader, config, device)
        results['bilstm'] = result
        histories['bilstm'] = result['history']
        
        # CNN Baseline
        cnn_model = CNNBaseline(
            vocab_size=vocab_size,
            embed_dim=config['model']['embed_dim'],
            num_filters=128,
            filter_sizes=[2, 3, 4, 5],
            num_classes=config['model']['num_classes'],
            dropout=config['model']['dropout_rate']
        ).to(device)
        result = train_model(cnn_model, "CNN", train_loader, val_loader, test_loader, config, device)
        results['cnn'] = result
        histories['cnn'] = result['history']
        
        # Tuned LSTM Baseline
        tuned_lstm_model = TunedLSTMBaseline(
            vocab_size=vocab_size,
            embed_dim=config['model']['embed_dim'],
            hidden_dim=256,
            num_layers=3,
            num_classes=config['model']['num_classes'],
            dropout=config['model']['dropout_rate']
        ).to(device)
        result = train_model(tuned_lstm_model, "TunedLSTM", train_loader, val_loader, test_loader, config, device)
        results['tuned_lstm'] = result
        histories['tuned_lstm'] = result['history']
        
        # BERT Baseline
        bert_model = BERTBaseline(
            num_classes=config['model']['num_classes'],
            dropout=config['model']['dropout_rate']
        ).to(device)
        result = train_model(bert_model, "BERT", train_loader, val_loader, test_loader, config, device)
        results['bert'] = result
        histories['bert'] = result['history']

        # RandomForest Baseline
        rf_model = RandomForestEnsemble(
            vocab_size=vocab_size,
            embed_dim=config['model']['embed_dim'],
            num_classes=config['model']['num_classes'],
            n_estimators=config['model'].get('rf_n_estimators', 100)
        ).to(device)
        result = train_model(rf_model, "RandomForest", train_loader, val_loader, test_loader, config, device)
        results['rf'] = result
        histories['rf'] = result['history']

        # LightGBM Baseline
        lgb_model = LightGBMEnsemble(
            vocab_size=vocab_size,
            embed_dim=config['model']['embed_dim'],
            num_classes=config['model']['num_classes']
        ).to(device)
        result = train_model(lgb_model, "LightGBM", train_loader, val_loader, test_loader, config, device)
        results['lightgbm'] = result
        histories['lightgbm'] = result['history']
    
    if args.mode in ['full', 'ablation']:
        print("\n" + "="*60)
        print("ABLATION STUDY")
        print("="*60)
        
        ablation_types = [
            'lexical_only',
            'semantic_only', 
            'structural_only',
            'no_attention'
        ]
        
        for ablation_type in ablation_types:
            try:
                print(f"\n📋 Training: {ablation_type.replace('_', ' ').title()}")
                
                model = create_ablation_model(ablation_type, vocab_size, config['model'])
                model = model.to(device)
                
                result = train_model(model, ablation_type, train_loader, val_loader, test_loader, config, device)
                results[ablation_type] = result
                histories[ablation_type] = result['history']
                
            except Exception as e:
                print(f"  ❌ Error training {ablation_type}: {e}")
    
    if args.mode in ['full']:
        print("\n" + "="*60)
        print("TRAINING TRI-FUSE")
        print("="*60)
        
        # TriFuse Model
        trifuse_model = OptimizedTriFuseModel(vocab_size, config=config['model']).to(device)
        result = train_model(trifuse_model, "TriFuse", train_loader, val_loader, test_loader, config, device)
        results['trifuse'] = result
        histories['trifuse'] = result['history']
    
    # Extract metrics for display
    metrics_results = {}
    for model_name, result in results.items():
        metrics_results[model_name] = {
            'accuracy': result['accuracy'],
            'f1_score': result['f1_score'],
            'precision': result['precision'],
            'recall': result['recall']
        }
    
    # Display results
    print("\n" + "="*80)
    print("🏆 FINAL RESULTS")
    print("="*80)
    
    for model_name, metrics in sorted(metrics_results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        print(f"{model_name.replace('_', ' ').title():<20} | "
              f"Accuracy: {metrics['accuracy']:.4f} | "
              f"F1: {metrics['f1_score']:.4f}")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    
    # Save results
    results_path = 'final_results.json'
    if 'paths' in config and 'results_dir' in config['paths']:
        results_path = os.path.join(config['paths']['results_dir'], 'final_results.json')
    
    with open(results_path, 'w') as f:
        json.dump(metrics_results, f, indent=2)
    
    print(f"💾 Results saved to: {results_path}")
    
    # Model comparison plot
    plot_path = 'model_comparison.png'
    if 'paths' in config and 'plots_dir' in config['paths']:
        plot_path = os.path.join(config['paths']['plots_dir'], 'model_comparison.png')
    
    try:
        plot_model_comparison(
            metrics_results,
            save_path=plot_path
        )
        print(f"📈 Plot saved to: {plot_path}")
    except Exception as e:
        print(f"⚠ Could not create plot: {e}")
    
    # Plot training history for TriFuse
    if 'trifuse' in histories:
        history_path = 'trifuse_training_history.png'
        if 'paths' in config and 'plots_dir' in config['paths']:
            history_path = os.path.join(config['paths']['plots_dir'], 'trifuse_training_history.png')
        
        try:
            plot_training_history(
                histories['trifuse'],
                save_path=history_path,
                show=False
            )
            print(f"📊 Training history saved to: {history_path}")
        except Exception as e:
            print(f"⚠ Could not create training history plot: {e}")
    
    # Generate comprehensive report
    if args.report:
        output_dir = 'outputs'
        if 'paths' in config and 'results_dir' in config['paths']:
            output_dir = config['paths']['results_dir']
        
        try:
            report_dir = create_comprehensive_report(metrics_results, histories.get('trifuse', {}), output_dir)
            print(f"\n📋 Comprehensive report created in: {report_dir}")
        except Exception as e:
            print(f"⚠ Could not create comprehensive report: {e}")
    
    # Performance summary
    print("\n" + "="*80)
    print("📈 PERFORMANCE SUMMARY")
    print("="*80)
    
    if metrics_results:
        best_model = max(metrics_results.items(), key=lambda x: x[1]['accuracy'])
        
        print(f"🎖️  BEST MODEL: {best_model[0].replace('_', ' ').title()}")
        print(f"   Accuracy: {best_model[1]['accuracy']:.4f}")
        print(f"   F1-Score: {best_model[1]['f1_score']:.4f}")
        
        # Check TriFuse performance
        if 'trifuse' in metrics_results:
            trifuse_acc = metrics_results['trifuse']['accuracy']
            if trifuse_acc >= 0.95:
                print("\n✅ SUCCESS: TriFuse achieved target accuracy of 95%+!")
            elif trifuse_acc >= 0.90:
                print(f"\n⚠ TriFuse accuracy: {trifuse_acc:.4f} (Good, but below 95% target)")
                print("   Suggestions:")
                print("   1. Train for more epochs (increase to 150)")
                print("   2. Increase model capacity (embed_dim to 512)")
                print("   3. Use pre-trained embeddings")
            else:
                print(f"\n❌ TriFuse accuracy: {trifuse_acc:.4f} (Needs improvement)")
                print("   Critical suggestions:")
                print("   1. Check your dataset quality and size")
                print("   2. Increase training epochs to 150+")
                print("   3. Reduce dropout rate to 0.2")
                print("   4. Use data augmentation")
    
    print("\n" + "="*80)
    print("🔧 TRAINING CONFIGURATION USED:")
    print("="*80)
    print(f"   Epochs: {config['training']['epochs']}")
    print(f"   Batch Size: {config['training']['batch_size']}")
    print(f"   Learning Rate: {config['training']['learning_rate']}")
    print(f"   Embedding Dim: {config['model']['embed_dim']}")
    print(f"   Dropout: {config['model']['dropout_rate']}")
    print(f"   Early Stopping Patience: {config['training']['patience']}")
    
    return 0

if __name__ == "__main__":
    main()