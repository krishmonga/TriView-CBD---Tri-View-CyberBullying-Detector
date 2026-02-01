"""
FIXED Ablation Models for TriFuse
All models now have consistent forward() signature
WITH FIXED IMPORTS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

# FIXED IMPORT: Try multiple ways to import models
try:
    # First try: Direct import (when running from src directory)
    from models import EnhancedLexicalView, EnhancedSemanticView, EnhancedStructuralView
    print("? Imported models directly")
except ImportError:
    try:
        # Second try: Relative import (when imported as module)
        from .models import EnhancedLexicalView, EnhancedSemanticView, EnhancedStructuralView
        print("? Imported models relatively")
    except ImportError:
        try:
            # Third try: Add parent directory to path
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            sys.path.insert(0, parent_dir)
            sys.path.insert(0, current_dir)
            from models import EnhancedLexicalView, EnhancedSemanticView, EnhancedStructuralView
            print("? Imported models with path adjustment")
        except ImportError as e:
            print(f"? Failed to import models: {e}")
            print("Creating dummy classes to avoid import error...")
            
            # Create dummy classes if import fails
            class DummyView(nn.Module):
                def __init__(self, *args, **kwargs):
                    super().__init__()
                    self.embedding = nn.Embedding(10000, 300)
                    self.output = nn.Linear(300, 256)
                
                def forward(self, x):
                    embedded = self.embedding(x)
                    return self.output(embedded.mean(dim=1))
            
            EnhancedLexicalView = DummyView
            EnhancedSemanticView = DummyView
            EnhancedStructuralView = DummyView

class LexicalOnlyModel(nn.Module):
    """Ablation: Only lexical view"""
    def __init__(self, vocab_size: int, embed_dim: int = 300, 
                 num_classes: int = 2, config: Dict = None):
        super().__init__()
        self.config = config or {}
        
        # Lexical view
        self.lexical_view = EnhancedLexicalView(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            dropout_rate=self.config.get('dropout_rate', 0.3),
            config=self.config
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.config.get('dropout_rate', 0.3)),
            nn.Linear(128, num_classes)
        )
        
        # Initialize classifier weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - FIXED: Only x parameter"""
        features = self.lexical_view(x)
        logits = self.classifier(features)
        return logits

class SemanticOnlyModel(nn.Module):
    """Ablation: Only semantic view"""
    def __init__(self, vocab_size: int, embed_dim: int = 300,
                 num_classes: int = 2, config: Dict = None):
        super().__init__()
        self.config = config or {}
        
        # Semantic view
        self.semantic_view = EnhancedSemanticView(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            dropout_rate=self.config.get('dropout_rate', 0.3),
            config=self.config
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.config.get('dropout_rate', 0.3)),
            nn.Linear(128, num_classes)
        )
        
        # Initialize classifier weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.semantic_view(x)
        logits = self.classifier(features)
        return logits

class StructuralOnlyModel(nn.Module):
    """Ablation: Only structural view"""
    def __init__(self, vocab_size: int, embed_dim: int = 300,
                 num_classes: int = 2, config: Dict = None):
        super().__init__()
        self.config = config or {}
        
        # Structural view
        self.structural_view = EnhancedStructuralView(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            dropout_rate=self.config.get('dropout_rate', 0.3),
            config=self.config
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.config.get('dropout_rate', 0.3)),
            nn.Linear(128, num_classes)
        )
        
        # Initialize classifier weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.structural_view(x)
        logits = self.classifier(features)
        return logits

class NoAttentionTriFuse(nn.Module):
    """Ablation: TriFuse without attention"""
    def __init__(self, vocab_size: int, embed_dim: int = 300,
                 num_classes: int = 2, config: Dict = None):
        super().__init__()
        self.config = config or {}
        
        # All three views
        self.lexical_view = EnhancedLexicalView(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            dropout_rate=self.config.get('dropout_rate', 0.3),
            config=self.config
        )
        
        self.semantic_view = EnhancedSemanticView(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            dropout_rate=self.config.get('dropout_rate', 0.3),
            config=self.config
        )
        
        self.structural_view = EnhancedStructuralView(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            dropout_rate=self.config.get('dropout_rate', 0.3),
            config=self.config
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.ReLU(),
            nn.Dropout(self.config.get('dropout_rate', 0.3)),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.config.get('dropout_rate', 0.3)),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Get features from all views
        lexical_features = self.lexical_view(x)
        semantic_features = self.semantic_view(x)
        structural_features = self.structural_view(x)
        
        # Concatenate
        combined = torch.cat([lexical_features, semantic_features, structural_features], dim=1)
        
        # Fusion
        fused = self.fusion(combined)
        
        # Classification
        logits = self.classifier(fused)
        return logits

# Also fix the TwoViewModel if you have it
class TwoViewModel(nn.Module):
    """Ablation: Two views combination"""
    def __init__(self, vocab_size: int, view1: str, view2: str,
                 embed_dim: int = 300, num_classes: int = 2, config: Dict = None):
        super().__init__()
        self.config = config or {}
        
        # View selection
        view_classes = {
            'lexical': EnhancedLexicalView,
            'semantic': EnhancedSemanticView,
            'structural': EnhancedStructuralView
        }
        
        self.view1 = view_classes[view1](
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            dropout_rate=self.config.get('dropout_rate', 0.3),
            config=self.config
        )
        
        self.view2 = view_classes[view2](
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            dropout_rate=self.config.get('dropout_rate', 0.3),
            config=self.config
        )
        
        # Attention
        self.attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.config.get('dropout_rate', 0.3)),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.attention.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Get features
        view1_features = self.view1(x)
        view2_features = self.view2(x)
        
        # Concatenate
        concatenated = torch.cat([view1_features, view2_features], dim=1)
        
        # Compute attention
        attention_weights = self.attention(concatenated)
        
        # Apply attention
        weighted_view1 = view1_features * attention_weights[:, 0:1]
        weighted_view2 = view2_features * attention_weights[:, 1:2]
        
        # Combine
        combined = torch.cat([weighted_view1, weighted_view2], dim=1)
        
        # Classification
        logits = self.classifier(combined)
        return logits

class SingleViewClassifier(nn.Module):
    """Classifier for single view models"""
    def __init__(self, view_model, num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.view = view_model
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Initialize classifier weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        features = self.view(x)
        logits = self.classifier(features)
        return logits

def create_ablation_model(ablation_type: str, vocab_size: int, config: Dict = None) -> nn.Module:
    """Factory function for ablation models"""
    config = config or {}
    embed_dim = config.get('embed_dim', 300)
    num_classes = config.get('num_classes', 2)
    
    if ablation_type == 'lexical_only':
        return LexicalOnlyModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_classes=num_classes,
            config=config
        )
    
    elif ablation_type == 'semantic_only':
        return SemanticOnlyModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_classes=num_classes,
            config=config
        )
    
    elif ablation_type == 'structural_only':
        return StructuralOnlyModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_classes=num_classes,
            config=config
        )
    
    elif ablation_type == 'no_attention':
        return NoAttentionTriFuse(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_classes=num_classes,
            config=config
        )
    
    elif ablation_type == 'lexical_semantic':
        return TwoViewModel(
            vocab_size=vocab_size,
            view1='lexical',
            view2='semantic',
            embed_dim=embed_dim,
            num_classes=num_classes,
            config=config
        )
    
    elif ablation_type == 'lexical_structural':
        return TwoViewModel(
            vocab_size=vocab_size,
            view1='lexical',
            view2='structural',
            embed_dim=embed_dim,
            num_classes=num_classes,
            config=config
        )
    
    elif ablation_type == 'semantic_structural':
        return TwoViewModel(
            vocab_size=vocab_size,
            view1='semantic',
            view2='structural',
            embed_dim=embed_dim,
            num_classes=num_classes,
            config=config
        )
    
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")


# Test the module
if __name__ == "__main__":
    print("Testing ablation_models.py...")
    
    # Create a simple config
    config = {
        'embed_dim': 300,
        'num_classes': 2,
        'dropout_rate': 0.3,
        'lexical_features': 130,
        'semantic_features': 309
    }
    
    # Test creating models
    vocab_size = 10000
    
    try:
        model = create_ablation_model('lexical_only', vocab_size, config)
        print(f"? Created lexical_only model with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        model = create_ablation_model('semantic_only', vocab_size, config)
        print(f"? Created semantic_only model with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        model = create_ablation_model('structural_only', vocab_size, config)
        print(f"? Created structural_only model with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        model = create_ablation_model('no_attention', vocab_size, config)
        print(f"? Created no_attention model with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        print("\n? All ablation models created successfully!")
        
        # Test forward pass
        test_input = torch.randint(0, 100, (2, 128))
        for ablation_type in ['lexical_only', 'semantic_only', 'structural_only', 'no_attention']:
            model = create_ablation_model(ablation_type, vocab_size, config)
            output = model(test_input)
            print(f"? {ablation_type}: Input shape {test_input.shape} -> Output shape {output.shape}")
            
    except Exception as e:
        print(f"? Error during testing: {e}")
        import traceback
        traceback.print_exc()