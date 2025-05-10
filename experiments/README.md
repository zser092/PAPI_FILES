# PaPI Experiments

This directory contains scripts to reproduce the experiments described in the PaPI papers.

## Experimental Conditions

The papers describe four main experimental conditions:

1. **Pre-Expansion**: Initial model training on the first task (SST-2)
2. **Baseline (No EWC)**: Sequential training without regularization
3. **Proposed (EWC, No Routing)**: Sequential training with Elastic Weight Consolidation but without the routing mechanism
4. **Proposed (EWC, With Routing)**: Full PaPI implementation with both EWC and the routing mechanism

## Running the Experiments

The `run_experiments.py` script automates the process of running all four experimental conditions and collecting the results.

```bash
# Navigate to the experiments directory
cd experiments

# Run all experiments
python run_experiments.py --output_dir ../results

# Run with a specific seed
python run_experiments.py --output_dir ../results --seed 123
```

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

## Experiment Details

Each experiment follows these steps:

1. **Model Initialization**: Initialize the PaPI model with RoBERTa-base and task-specific adapters
2. **Sequential Training**: Train on SST-2, then Emotion, then MNLI
3. **EWC Application**: For conditions with EWC, compute the Fisher Information Matrix after training on SST-2
4. **Routing Training**: For conditions with routing, train the routing classifier on task-specific data
5. **Evaluation**: Evaluate performance on all tasks after each training phase
6. **Energy Tracking**: Track energy consumption and estimate CO2 emissions

## Output

The script generates the following outputs in the specified output directory:

1. **Individual experiment results**: Each experiment has its own subdirectory with:
   - Trained model weights
   - Results CSV file with metrics

2. **Comparative visualizations**:
   - Task accuracies across experiments
   - Forgetting rates across experiments
   - Energy consumption comparison
   - CO2 emissions comparison

3. **Summary tables**:
   - CSV summary of all metrics
   - Markdown table for easy inclusion in documentation

## Customizing Experiments

You can modify the `run_experiments.py` script to customize the experiments:

- Change hyperparameters like learning rate or batch size
- Add new experimental conditions
- Modify the evaluation metrics
- Change the sequence of tasks

## References

For more details on the experimental setup and methodology, refer to the original papers:
- "PaPI: Learning More, Wasting Less"
- "PaPI: Pathway-based Progressive Inference for Energy-Efficient Continual Learning"