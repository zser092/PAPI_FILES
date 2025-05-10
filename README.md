# PaPI: Pathway-based Progressive Inference

This repository contains the implementation of PaPI (Pathway-based Progressive Inference), a framework for energy-efficient continual learning in Natural Language Processing (NLP). The framework addresses the dual challenges of preventing catastrophic forgetting while maintaining energy efficiency, particularly in resource-constrained environments.

## Overview

PaPI combines several key components:
- **RoBERTa-base** as the foundation model
- **Lightweight adapters** for parameter-efficient fine-tuning (1-3% of parameters)
- **Elastic Weight Consolidation (EWC)** for preventing catastrophic forgetting
- **Softmax-based routing mechanism** for dynamic task selection

The framework is designed to achieve high task performance while significantly reducing computational costs and environmental impact, using only 2% of the energy required for full fine-tuning.

## Repository Structure

```
repository/
├── code/                      # Source code organized by experiment/functionality
│   ├── models/                # Model implementations
│   │   ├── adapters.py        # Implementation of adapter modules
│   │   ├── papi_model.py      # Main PaPI model implementation
│   │   └── routing_classifier.py # Routing mechanism implementation
│   ├── losses/
│   │   └── ewc_regularization.py # EWC regularization implementation
│   ├── utils/
│   │   └── routing_utils.py   # Utility functions for routing and evaluation
│   └── train_papi.py          # Training script for the PaPI model
├── data/                      # Instructions for data access and preparation
├── experiments/               # Scripts to reproduce specific experiments
│   ├── run_experiments.py     # Script to run all experiments
│   └── README.md              # Experiment-specific documentation
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment specification
└── README.md                  # This file
```

## Installation

### Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/papi.git
cd papi

# Create a virtual environment
python -m venv papi-env
source papi-env/bin/activate  # On Windows: papi-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Using conda

```bash
# Clone the repository
git clone https://github.com/yourusername/papi.git
cd papi

# Create and activate conda environment
conda env create -f environment.yml
conda activate papi-env
```

## Data Preparation

The PaPI framework was evaluated on three NLP benchmark datasets:

### 1. SST-2 (Stanford Sentiment Treebank)
- Binary sentiment classification task (positive/negative)
- Part of the GLUE benchmark
- Automatically downloaded via HuggingFace datasets library

### 2. Emotion
- Emotion classification task with 6 classes (anger, fear, joy, love, sadness, surprise)
- Available through HuggingFace datasets

### 3. MNLI (Multi-Genre Natural Language Inference)
- Natural language inference task (entailment, contradiction, neutral)
- Part of the GLUE benchmark
- Automatically downloaded via HuggingFace datasets library

For detailed instructions on data preparation, see the [data/README.md](data/README.md) file.

## Running Experiments

The PaPI framework includes four main experimental conditions:

1. **Pre-Expansion**: Initial model training on the first task (SST-2)
2. **Baseline (No EWC)**: Sequential training without regularization
3. **Proposed (EWC, No Routing)**: Sequential training with Elastic Weight Consolidation but without the routing mechanism
4. **Proposed (EWC, With Routing)**: Full PaPI implementation with both EWC and the routing mechanism

To run all experiments:

```bash
# Navigate to the experiments directory
cd experiments

# Run all experiments
python run_experiments.py --output_dir ../results

# Run with a specific seed
python run_experiments.py --output_dir ../results --seed 123
```

For detailed instructions on running experiments, see the [experiments/README.md](experiments/README.md) file.

## Training a PaPI Model

To train a PaPI model on a sequence of tasks:

```bash
python code/train_papi.py --output_dir ./output --use_routing --freeze_base
```

### Command-line Arguments

- `--output_dir`: Output directory (default: "./output")
- `--model_name`: Pretrained model name (default: "roberta-base")
- `--adapter_dim`: Adapter bottleneck dimension (default: 64)
- `--use_routing`: Use routing mechanism (flag)
- `--routing_temp`: Routing temperature (default: 1.0)
- `--freeze_base`: Freeze base model parameters (flag)
- `--ewc_lambda`: EWC regularization strength (default: 1000.0)
- `--fisher_samples`: Number of samples for Fisher computation (default: 500)
- `--batch_size`: Batch size (default: 32)
- `--num_epochs`: Number of epochs (default: 3)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--max_seq_length`: Maximum sequence length (default: 128)
- `--seed`: Random seed (default: 42)

## Key Components

### Adapter Modules

Adapters are small, parameter-efficient modules inserted into transformer models that:
- Enable efficient transfer of pre-trained models to new tasks
- Minimize retraining of base model parameters (only 1-3% of parameters are updated)
- Are implemented as feed-forward layers that project features to a smaller dimension and back

In the PaPI framework, adapters are used to:
- Enable task-specific customization
- Support efficient fine-tuning
- Facilitate learning across multiple tasks without catastrophic forgetting

### Elastic Weight Consolidation (EWC)

EWC is a regularization technique used to prevent catastrophic forgetting by:
1. Identifying and protecting crucial network parameters that are important for previously learned tasks
2. "Elastically" anchoring important weights to prevent complete overwriting
3. Using the Fisher Information Matrix to quantify parameter importance
4. Adding a regularization term to the loss function that penalizes changes to important parameters

The EWC loss is computed as:
```
L_reg = (λ/2) * sum_i F_i * (θ_i - θ*_i)²
```
where:
- λ is the regularization strength (fisher_alpha)
- F_i is the Fisher Information for parameter θ_i
- θ_i is the current value of parameter i
- θ*_i is the optimal value of parameter i for previous tasks

### Routing Mechanism

The softmax-based routing mechanism dynamically allocates input samples to appropriate tasks, enhancing computational efficiency and adaptability. The routing probability is computed as:

```
P(t|x) = exp(w_t^T * h(x) / temperature) / sum_k(exp(w_k^T * h(x) / temperature))
```

where:
- w_t is the weight vector for task t
- h(x) is the feature representation of input x
- temperature controls the sharpness of the distribution

## Expected Results

Based on the papers, you should expect the following results:

### Task Performance

| Condition | SST-2 Accuracy | Emotion Accuracy | MNLI Accuracy | Routing Accuracy |
|-----------|----------------|------------------|---------------|------------------|
| Pre-Expansion | 93.89% | - | - | - |
| Baseline (No EWC) | 49.44% | 62.20% | - | - |
| PaPI (EWC, No Routing) | 93.00% | 92.60% | 78.00% | - |
| PaPI (EWC, With Routing) | 90.50% | 89.50% | 76.00% | ~90% |

### Environmental Impact

| Model/Method | Energy (kWh) | CO2 Emissions (g) | Notes |
|--------------|--------------|-------------------|-------|
| Full Fine-Tuning | 4.608 | 1,843 | 3 epochs, 400 gCO2/kWh |
| PaPI | 0.09216 | 37 | 2% of full fine-tuning |

## Citation

If you use this code in your research, please cite the original papers:

```
This code accompanies our ongoing work on PaPI. Citations are provisional and will be finalized upon paper acceptance
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
