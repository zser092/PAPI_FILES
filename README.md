# PaPI: Pathway-based Progressive Inference

This repository contains the implementation of PaPI (Pathway-based Progressive Inference), a framework for energy-efficient continual learning in Natural Language Processing (NLP). The framework addresses the dual challenges of preventing catastrophic forgetting while maintaining energy efficiency, particularly in resource-constrained environments.

## Overview

PaPI combines several key components:
- **RoBERTa-base** as the foundation model
- **Lightweight adapters** for parameter-efficient fine-tuning
- **Elastic Weight Consolidation (EWC)** for preventing catastrophic forgetting
- **Softmax-based routing mechanism** for dynamic task selection

The framework is designed to achieve high task performance while significantly reducing computational costs and environmental impact.

## Directory Structure

```
repository/
├── code/
│   ├── models/
│   │   ├── adapters.py          # Implementation of adapter modules
│   │   ├── papi_model.py        # Main PaPI model implementation
│   │   └── routing_classifier.py # Routing mechanism implementation
│   ├── losses/
│   │   └── ewc_regularization.py # EWC regularization implementation
│   ├── utils/
│   │   └── routing_utils.py     # Utility functions for routing and evaluation
│   └── train_papi.py            # Training script for the PaPI model
├── data/                        # Instructions for data access (datasets loaded via HuggingFace)
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment specification
└── README.md                    # This file
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

## Usage

### Training the PaPI model

The `train_papi.py` script demonstrates how to train the PaPI model on a sequence of tasks (SST-2, Emotion, and MNLI) using the Elastic Weight Consolidation (EWC) regularization technique to prevent catastrophic forgetting.

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

## Datasets

The implementation uses the following datasets, which are automatically downloaded via the HuggingFace `datasets` library:

1. **SST-2 (Stanford Sentiment Treebank)**
   - Binary sentiment classification task (positive/negative)
   - Part of the GLUE benchmark

2. **Emotion**
   - Emotion classification task with 6 classes
   - Available through HuggingFace datasets

3. **MNLI (Multi-Genre Natural Language Inference)**
   - Natural language inference task
   - Part of the GLUE benchmark

## Key Components

### Adapter Modules

Adapters are small, parameter-efficient modules inserted into transformer models that:
- Enable efficient transfer of pre-trained models to new tasks
- Minimize retraining of base model parameters
- Are typically implemented as feed-forward layers that project features to a smaller dimension and back

### Elastic Weight Consolidation (EWC)

EWC is a regularization technique used to prevent catastrophic forgetting by:
1. Identifying and protecting crucial network parameters that are important for previously learned tasks
2. "Elastically" anchoring important weights to prevent complete overwriting
3. Using the Fisher Information Matrix to quantify parameter importance
4. Adding a regularization term to the loss function that penalizes changes to important parameters

### Routing Mechanism

The softmax-based routing mechanism dynamically allocates input samples to appropriate tasks, enhancing computational efficiency and adaptability. The routing probability is computed as:

```
P(t|x) = exp(w_t^T * h(x)) / sum_k(exp(w_k^T * h(x)))
```

where `w_t` is the weight vector for task `t` and `h(x)` is the feature representation of input `x`.

## Results

As reported in the papers, PaPI achieves competitive accuracies:
- SST-2: 93.00%
- Emotion: 92.60%
- MNLI: 78.00%

While significantly reducing energy consumption:
- Full Fine-Tuning: 4.608 kWh, 1,843g CO2
- PaPI: 0.09216 kWh, 37g CO2 (2% of full fine-tuning)

## Citation

If you use this code in your research, please cite the original papers:

```
@inproceedings{gaurav2025papi,
  title={PaPI: Learning More, Wasting Less},
  author={Gaurav, Suyash and Heikkonen, Jukka and Chaudhary, Jatin},
  booktitle={Proceedings of the Conference},
  year={2025}
}

@inproceedings{gaurav2025pathway,
  title={PaPI: Pathway-based Progressive Inference for Energy-Efficient Continual Learning},
  author={Gaurav, Suyash and Heikkonen, Jukka and Chaudhary, Jatin},
  booktitle={39th Conference on Neural Information Processing Systems (NeurIPS 2025)},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.