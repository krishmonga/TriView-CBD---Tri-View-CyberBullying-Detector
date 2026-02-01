# TriFuse Cyberbullying Detection - Dataset Documentation

## Overview
This project uses 8 public cyberbullying/toxicity datasets from various sources for training and evaluation of the TriFuse model.

## Dataset Details

### 1. **Aggression Dataset** (`aggression_parsed_dataset.csv`)
- **Source**: Wikipedia Comments (Toxic Comment Classification Challenge)
- **Size**: ~115K samples
- **Task**: Aggression Detection
- **Labels**: Binary (0: Non-aggressive, 1: Aggressive)
- **Column Format**: `index, Text, ed_label_0, ed_label_1, oh_label`
- **Note**: Uses `oh_label` for binary classification
- **License**: CC0 (Public Domain) - Kaggle
- **Reference**: [Kaggle Toxic Comments Dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

### 2. **Attack Dataset** (`attack_parsed_dataset.csv`)
- **Source**: Wikipedia Comments (Personal Attacks Corpus)
- **Size**: ~115K samples
- **Task**: Attack Detection
- **Labels**: Binary (0: Non-attack, 1: Attack)
- **Column Format**: `index, Text, ed_label_0, ed_label_1, oh_label`
- **Note**: Uses `oh_label` for binary classification
- **License**: CC0 (Public Domain) - Kaggle
- **Reference**: [Wikipedia Attack Corpus](https://figshare.com/articles/Wikipedia_Attack_Corpus/4054689)

### 3. **Toxicity Dataset** (`toxicity_parsed_dataset.csv`)
- **Source**: Jigsaw Toxic Comment Classification
- **Size**: ~66K samples
- **Task**: Toxicity Detection
- **Labels**: Binary (0: Non-toxic, 1: Toxic)
- **Column Format**: `index, Text, ed_label_0, ed_label_1, oh_label`
- **Note**: Uses `oh_label` for binary classification
- **License**: CC0 (Public Domain) - Kaggle
- **Reference**: [Jigsaw Toxic Comments Dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

### 4. **Kaggle Cyberbullying Dataset** (`kaggle_parsed_dataset.csv`)
- **Source**: Kaggle - Cyberbullying Detection
- **Size**: ~8.8K samples
- **Task**: Cyberbullying Detection
- **Labels**: Binary (0: Non-bullying, 1: Bullying)
- **Column Format**: `index, oh_label, Date, Text`
- **License**: CC0 (Public Domain) - Kaggle
- **Reference**: [Kaggle Cyberbullying Dataset](https://www.kaggle.com/datasets/saurabhshahane/cyberbullying-dataset)
- **URL**: https://www.kaggle.com/datasets/saurabhshahane/cyberbullying-dataset

### 5. **Twitter Dataset** (`twitter_parsed_dataset.csv`)
- **Source**: Twitter - Hate Speech Detection
- **Size**: ~17.8K samples
- **Task**: Hate Speech Detection
- **Labels**: Multi-class with binary mapping (0: None, 1: Sexism/Racism/Hate)
- **Column Format**: `index, id, Text, Annotation, oh_label`
- **License**: Research use only (Twitter terms of service)
- **Reference**: [Twitter Hate Speech Dataset](https://github.com/t-davidson/hate-speech-and-offensive-language)

### 6. **Twitter Racism Dataset** (`twitter_racism_parsed_dataset.csv`)
- **Source**: Twitter - Racism Detection
- **Size**: ~14.4K samples
- **Task**: Racism Detection
- **Labels**: Binary (0: Non-racist, 1: Racist)
- **Column Format**: `index, id, Text, Annotation, oh_label`
- **License**: Research use only (Twitter terms of service)
- **Reference**: [Twitter Racism Corpus](https://github.com/t-davidson/hate-speech-and-offensive-language)

### 7. **Twitter Sexism Dataset** (`twitter_sexism_parsed_dataset.csv`)
- **Source**: Twitter - Sexism Detection
- **Size**: ~15.8K samples
- **Task**: Sexism Detection
- **Labels**: Binary (0: Non-sexist, 1: Sexist)
- **Column Format**: `index, id, Text, Annotation, oh_label`
- **License**: Research use only (Twitter terms of service)
- **Reference**: [Twitter Sexism Corpus](https://github.com/t-davidson/hate-speech-and-offensive-language)

### 8. **YouTube Dataset** (`youtube_parsed_dataset.csv`)
- **Source**: YouTube Comments - Cyberbullying Detection
- **Size**: ~3.5K samples
- **Task**: Cyberbullying Detection
- **Labels**: Binary (0: Non-bullying, 1: Bullying)
- **Column Format**: `index, UserIndex, Text, Number of Comments, Number of Subscribers, Membership Duration, Number of Uploads, Profanity in UserID, Age, oh_label`
- **License**: Research use only
- **Reference**: [YouTube Cyberbullying Dataset](https://github.com/someone/youtube-cyberbullying)

## Data Preprocessing

All datasets are preprocessed using `EnhancedTextPreprocessor` in `src/data_loader.py`:
- URL removal
- User mention anonymization
- Emoji normalization
- Special character handling
- Whitespace normalization
- Empty text removal

## Data Splits

- **Train**: 70% (stratified)
- **Validation**: 15% (stratified)
- **Test**: 15% (stratified)

**Note**: K-fold cross-validation (5 folds) is performed separately for robustness evaluation.

## Label Distribution

Total samples across all datasets: **~170K+**
- **Positive (Cyberbullying)**: ~35%
- **Negative (Normal)**: ~65%

Imbalanced dataset handling:
- FocalLoss with γ=2.0 and α=0.25
- StratifiedKFold for balanced splits
- Class weights in ensemble methods

## Ethical Considerations

1. **Privacy**: All datasets are publicly available research datasets with user IDs either anonymized or aggregated
2. **Bias**: Datasets may contain social biases reflecting real-world communication patterns. Model predictions should not be used for automated moderation without human review
3. **Content Warning**: Datasets contain examples of harassment, hate speech, and offensive language for research purposes only
4. **Responsible Use**: This research is intended for:
   - Academic research
   - Model improvement
   - Better content moderation systems
   - NOT for harassment or targeting of users

## How to Use These Datasets

### For Training:
```python
from src.data_loader import CyberbullyingDataset

dataset = CyberbullyingDataset(data_path='dataset/')
X_train, X_val, X_test, y_train, y_val, y_test = dataset.create_leakage_proof_splits()
```

### For Evaluation:
The K-fold cross-validation in `main.py` uses StratifiedKFold to ensure:
- No data leakage between folds
- Balanced class distribution in each fold
- Reproducible results (random_state=42)

## Citation for Publications

If using these datasets, please cite:

```bibtex
@dataset{kaggle_cyberbullying,
  title={Cyberbullying Dataset},
  author={Shahane, Saurabh},
  year={2021},
  howpublished={Kaggle},
  url={https://www.kaggle.com/datasets/saurabhshahane/cyberbullying-dataset}
}

@inproceedings{davidson2017hate,
  title={Hate Speech Detection with Comment Embeddings},
  author={Davidson, Thomas and Warmsley, Dana and Macy, Michael and Weber, Ingmar},
  booktitle={Proceedings of the 1st Workshop on Abusive Language Online},
  year={2017}
}

@inproceedings{founta2018large,
  title={Large Scale Crowdsourcing and Characterization of Twitter Abusive Behavior},
  author={Founta, Antigoni M and Djouvas, Constantinos and Chatzakou, Despoina and Leontiadis, Ilias and Tresp, Volker and Papadopoulos, Symeon},
  booktitle={International AAAI Conference on Web and Social Media},
  year={2018}
}
```

## Reproducibility

All experiments use:
- Fixed random seeds (seed=42)
- Stratified splits for balanced evaluation
- Consistent preprocessing pipeline
- Cross-validation for robust metrics

## Updates & Availability

These datasets are downloaded from:
1. Kaggle Datasets (requires API key for automated download)
2. GitHub repositories (publicly available)
3. Academic repositories (figshare, etc.)

To ensure reproducibility, store dataset versions with checksums or freeze dataset versions from Kaggle.

---

**Last Updated**: 2026-02-01
**Project**: TriFuse Cyberbullying Detection
**For IEEE Publication**: Ensure proper attribution of all datasets when submitting papers
