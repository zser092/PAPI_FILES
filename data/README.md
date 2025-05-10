# Data Access and Preparation

This directory contains instructions for accessing and preparing the datasets used in the PaPI framework experiments.

## Datasets

The PaPI framework was evaluated on three NLP benchmark datasets:

### 1. SST-2 (Stanford Sentiment Treebank)

- **Description**: Binary sentiment classification task (positive/negative)
- **Source**: Part of the GLUE (General Language Understanding Evaluation) benchmark
- **Size**: 
  - Training: 67,349 samples
  - Validation: 872 samples
- **Access**: Automatically downloaded via HuggingFace datasets library
  ```python
  from datasets import load_dataset
  dataset = load_dataset("glue", "sst2")
  ```

### 2. Emotion

- **Description**: Emotion classification task with 6 classes (anger, fear, joy, love, sadness, surprise)
- **Source**: Available through HuggingFace datasets
- **Size**:
  - Training: 16,000 samples
  - Validation: 2,000 samples
- **Access**: Automatically downloaded via HuggingFace datasets library
  ```python
  from datasets import load_dataset
  dataset = load_dataset("emotion")
  ```

### 3. MNLI (Multi-Genre Natural Language Inference)

- **Description**: Natural language inference task (entailment, contradiction, neutral)
- **Source**: Part of the GLUE benchmark
- **Size**:
  - Training: 392,702 samples
  - Validation (matched): 9,815 samples
  - Validation (mismatched): 9,832 samples
- **Access**: Automatically downloaded via HuggingFace datasets library
  ```python
  from datasets import load_dataset
  dataset = load_dataset("glue", "mnli")
  ```

## Data Preprocessing

The datasets are preprocessed in the training script (`code/train_papi.py`) using the following steps:

1. **Tokenization**: Using RoBERTa tokenizer
2. **Truncation/Padding**: Sequences are truncated or padded to a maximum length (default: 128 tokens)
3. **Special Handling for MNLI**: Premise and hypothesis are concatenated with a separator token

Example preprocessing code:
```python
def preprocess_function(examples):
    # For SST-2 and Emotion
    if "sentence" in examples:
        texts = examples["sentence"]
    elif "text" in examples:
        texts = examples["text"]
    # For MNLI
    else:
        texts = [f"{premise} {tokenizer.sep_token} {hypothesis}" 
                for premise, hypothesis in zip(examples["premise"], examples["hypothesis"])]
    
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=128,
    )
```

## Data Sampling for EWC

For the Elastic Weight Consolidation (EWC) regularization, a subset of samples is used to compute the Fisher Information Matrix. As mentioned in the paper, 500 samples from the SST-2 dataset were used for this purpose.

```python
# Sample 500 examples from SST-2 for Fisher computation
fisher_sample_size = 500
indices = torch.randperm(len(sst2_dataset))[:fisher_sample_size]
fisher_dataset = torch.utils.data.Subset(sst2_dataset, indices)
```

## Sequential Task Learning

In the PaPI framework, the tasks are learned sequentially in the following order:
1. SST-2 (sentiment analysis)
2. Emotion (emotion classification)
3. MNLI (natural language inference)

This order is maintained in the training script to replicate the experimental setup from the papers.