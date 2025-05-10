"""
Script to reproduce the experiments from the PaPI papers.

This script runs the experiments described in the papers, including:
1. Pre-Expansion (training only on SST-2)
2. Baseline (No EWC) - Sequential training without regularization
3. Proposed (EWC, No Routing) - Sequential training with EWC but without routing
4. Proposed (EWC, With Routing) - Full PaPI implementation with both EWC and routing
"""

import os
import argparse
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run PaPI experiments")
    parser.add_argument("--output_dir", type=str, default="./experiment_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def run_experiment(experiment_name, output_dir, use_ewc=True, use_routing=True, ewc_lambda=1000.0, seed=42):
    """Run a single experiment."""
    # Create experiment directory
    experiment_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Build command
    cmd = [
        "python", "../code/train_papi.py",
        "--output_dir", experiment_dir,
        "--freeze_base",
        "--seed", str(seed),
    ]
    
    # Add EWC if specified
    if use_ewc:
        cmd.extend(["--ewc_lambda", str(ewc_lambda)])
    else:
        cmd.extend(["--ewc_lambda", "0.0"])  # Disable EWC
    
    # Add routing if specified
    if use_routing:
        cmd.append("--use_routing")
    
    # Run command
    print(f"\n=== Running experiment: {experiment_name} ===")
    print(f"Command: {' '.join(cmd)}")
    
    # Execute the command and capture output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    
    # Print output in real-time
    stdout, stderr = process.communicate()
    print(stdout)
    if stderr:
        print(f"Errors: {stderr}")
    
    # Check if experiment was successful
    if process.returncode != 0:
        print(f"Experiment {experiment_name} failed with return code {process.returncode}")
        return None
    
    # Parse results
    results = parse_results(stdout)
    results["experiment_name"] = experiment_name
    
    # Save results
    results_file = os.path.join(experiment_dir, "results.csv")
    pd.DataFrame([results]).to_csv(results_file, index=False)
    
    return results


def parse_results(output):
    """Parse experiment results from output."""
    results = {}
    
    # Extract accuracies
    if "Task 0 Accuracy:" in output:
        results["sst2_accuracy"] = float(output.split("Task 0 Accuracy:")[1].split()[0])
    
    if "Task 1 Accuracy:" in output:
        results["emotion_accuracy"] = float(output.split("Task 1 Accuracy:")[1].split()[0])
    
    if "Task 2 Accuracy:" in output:
        results["mnli_accuracy"] = float(output.split("Task 2 Accuracy:")[1].split()[0])
    
    # Extract transfer metrics
    if "Backward Transfer:" in output:
        results["backward_transfer"] = float(output.split("Backward Transfer:")[1].split()[0])
    
    if "Forward Transfer:" in output:
        results["forward_transfer"] = float(output.split("Forward Transfer:")[1].split()[0])
    
    # Extract forgetting rates
    if "Forgetting Rate for sst2:" in output:
        results["sst2_forgetting_rate"] = float(output.split("Forgetting Rate for sst2:")[1].split()[0])
    
    if "Forgetting Rate for emotion:" in output:
        results["emotion_forgetting_rate"] = float(output.split("Forgetting Rate for emotion:")[1].split()[0])
    
    # Extract energy consumption
    if "Estimated Energy:" in output:
        results["energy_kwh"] = float(output.split("Estimated Energy:")[1].split()[0])
    
    if "Estimated CO2 Emissions:" in output:
        results["co2_emissions_g"] = float(output.split("Estimated CO2 Emissions:")[1].split()[0])
    
    return results


def plot_results(results_list, output_dir):
    """Plot experiment results."""
    # Create a DataFrame from results
    df = pd.DataFrame(results_list)
    
    # Plot task accuracies
    plt.figure(figsize=(12, 6))
    
    # Set width of bars
    barWidth = 0.25
    
    # Set position of bars on X axis
    r1 = np.arange(len(df))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    # Create bars
    plt.bar(r1, df["sst2_accuracy"], width=barWidth, label="SST-2")
    plt.bar(r2, df["emotion_accuracy"], width=barWidth, label="Emotion")
    plt.bar(r3, df["mnli_accuracy"], width=barWidth, label="MNLI")
    
    # Add labels and title
    plt.xlabel("Experiment")
    plt.ylabel("Accuracy")
    plt.title("Task Accuracies Across Experiments")
    plt.xticks([r + barWidth for r in range(len(df))], df["experiment_name"], rotation=45)
    plt.legend()
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "task_accuracies.png"))
    
    # Plot forgetting rates
    plt.figure(figsize=(10, 6))
    
    # Set position of bars on X axis
    r1 = np.arange(len(df))
    r2 = [x + barWidth for x in r1]
    
    # Create bars
    plt.bar(r1, df["sst2_forgetting_rate"], width=barWidth, label="SST-2")
    plt.bar(r2, df["emotion_forgetting_rate"], width=barWidth, label="Emotion")
    
    # Add labels and title
    plt.xlabel("Experiment")
    plt.ylabel("Forgetting Rate")
    plt.title("Forgetting Rates Across Experiments")
    plt.xticks([r + barWidth/2 for r in range(len(df))], df["experiment_name"], rotation=45)
    plt.legend()
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "forgetting_rates.png"))
    
    # Plot energy consumption
    plt.figure(figsize=(10, 6))
    plt.bar(df["experiment_name"], df["energy_kwh"], color="green")
    plt.xlabel("Experiment")
    plt.ylabel("Energy Consumption (kWh)")
    plt.title("Energy Consumption Across Experiments")
    plt.xticks(rotation=45)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "energy_consumption.png"))
    
    # Plot CO2 emissions
    plt.figure(figsize=(10, 6))
    plt.bar(df["experiment_name"], df["co2_emissions_g"], color="brown")
    plt.xlabel("Experiment")
    plt.ylabel("CO2 Emissions (g)")
    plt.title("CO2 Emissions Across Experiments")
    plt.xticks(rotation=45)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "co2_emissions.png"))
    
    # Create a summary table
    summary = df[["experiment_name", "sst2_accuracy", "emotion_accuracy", "mnli_accuracy", 
                 "sst2_forgetting_rate", "emotion_forgetting_rate", 
                 "energy_kwh", "co2_emissions_g"]]
    
    # Save summary table
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    
    # Create a markdown table for the README
    with open(os.path.join(output_dir, "results_table.md"), "w") as f:
        f.write("# Experiment Results\n\n")
        f.write("## Task Accuracies\n\n")
        f.write("| Experiment | SST-2 | Emotion | MNLI |\n")
        f.write("|------------|-------|---------|------|\n")
        for _, row in df.iterrows():
            f.write(f"| {row['experiment_name']} | {row['sst2_accuracy']:.2%} | {row['emotion_accuracy']:.2%} | {row['mnli_accuracy']:.2%} |\n")
        
        f.write("\n## Forgetting Rates\n\n")
        f.write("| Experiment | SST-2 | Emotion |\n")
        f.write("|------------|-------|--------|\n")
        for _, row in df.iterrows():
            f.write(f"| {row['experiment_name']} | {row['sst2_forgetting_rate']:.4f} | {row['emotion_forgetting_rate']:.4f} |\n")
        
        f.write("\n## Energy Consumption and CO2 Emissions\n\n")
        f.write("| Experiment | Energy (kWh) | CO2 Emissions (g) |\n")
        f.write("|------------|-------------|-------------------|\n")
        for _, row in df.iterrows():
            f.write(f"| {row['experiment_name']} | {row['energy_kwh']:.6f} | {row['co2_emissions_g']:.2f} |\n")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define experiments
    experiments = [
        {
            "name": "Pre-Expansion",
            "use_ewc": False,
            "use_routing": False,
            "ewc_lambda": 0.0,
        },
        {
            "name": "Baseline_No_EWC",
            "use_ewc": False,
            "use_routing": False,
            "ewc_lambda": 0.0,
        },
        {
            "name": "Proposed_EWC_No_Routing",
            "use_ewc": True,
            "use_routing": False,
            "ewc_lambda": 1000.0,
        },
        {
            "name": "Proposed_EWC_With_Routing",
            "use_ewc": True,
            "use_routing": True,
            "ewc_lambda": 1000.0,
        },
    ]
    
    # Run experiments
    results_list = []
    for experiment in experiments:
        result = run_experiment(
            experiment_name=experiment["name"],
            output_dir=output_dir,
            use_ewc=experiment["use_ewc"],
            use_routing=experiment["use_routing"],
            ewc_lambda=experiment["ewc_lambda"],
            seed=args.seed,
        )
        if result is not None:
            results_list.append(result)
    
    # Plot and save results
    if results_list:
        plot_results(results_list, output_dir)
        print(f"\nExperiment results saved to {output_dir}")
    else:
        print("\nNo valid results to plot.")


if __name__ == "__main__":
    main()