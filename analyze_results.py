import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def load_pickle(file_path):
    print(f"Loading: {file_path}")
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def analyze_characteristics(data, name):
    print(f"\n=== Analysis for {name} ===")
    
    # Print mean values for each characteristic
    for key in sorted(data.keys()):
        values = np.array(data[key])
        
        # Count NaN values
        nan_count = np.isnan(values).sum()
        total_count = len(values)
        
        # Filter out NaN values for statistics
        valid_values = values[~np.isnan(values)]
        
        if len(valid_values) > 0:
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)
            print(f"{key:15s}: mean = {mean_val:.4f}, std = {std_val:.4f}, valid values = {len(valid_values)}/{total_count}")
        else:
            print(f"{key:15s}: ALL NaN VALUES ({total_count} total)")

def plot_distributions(train_data, test_data, adv_test_data, save_dir='characteristic_plots'):
    os.makedirs(save_dir, exist_ok=True)
    characteristics = sorted(train_data.keys())
    
    for char in characteristics:
        # Get non-NaN values for each dataset
        train_vals = np.array(train_data[char])[~np.isnan(train_data[char])]
        test_vals = np.array(test_data[char])[~np.isnan(test_data[char])]
        adv_test_vals = np.array(adv_test_data[char])[~np.isnan(adv_test_data[char])]
        
        # Skip if all values are NaN
        if len(train_vals) == 0 and len(test_vals) == 0 and len(adv_test_vals) == 0:
            print(f"Skipping plot for {char} - all values are NaN")
            continue
        
        plt.figure(figsize=(12, 6))
        
        # Plot only if we have valid values
        if len(train_vals) > 0:
            plt.hist(train_vals, alpha=0.4, label=f'Train (valid: {len(train_vals)})', bins=20, color='blue')
        if len(test_vals) > 0:
            plt.hist(test_vals, alpha=0.4, label=f'Test (valid: {len(test_vals)})', bins=20, color='green')
        if len(adv_test_vals) > 0:
            plt.hist(adv_test_vals, alpha=0.4, label=f'Adversarial Test (valid: {len(adv_test_vals)})', bins=20, color='red')
        
        plt.title(f'Distribution of {char}', fontsize=12, pad=20)
        plt.xlabel('Value', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.savefig(f'{save_dir}/{char.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()

def inspect_pickle_contents(data, name):
    print(f"\n=== Inspecting {name} ===")
    for key in sorted(data.keys()):
        values = np.array(data[key])
        print(f"\nKey: {key}")
        print(f"Shape: {values.shape}")
        print(f"NaN count: {np.isnan(values).sum()}")
        print(f"First few values: {values[:5]}")
        if not np.all(np.isnan(values)):
            print(f"Min: {np.nanmin(values)}")
            print(f"Max: {np.nanmax(values)}")

def main():
    # Set the correct paths
    base_dir = './DistriBlock_data'
    cw_dir = os.path.join(base_dir, 'CW')
    
    # Load all pickle files
    train_data = load_pickle(os.path.join(base_dir, 'train.pickle'))
    test_data = load_pickle(os.path.join(base_dir, 'test.pickle'))
    adv_test_data = load_pickle(os.path.join(cw_dir, 'adv_test.pickle'))
    
    # Inspect the contents of each pickle file
    print("\n=== Detailed Inspection of Pickle Files ===")
    inspect_pickle_contents(train_data, "Training Data")
    inspect_pickle_contents(test_data, "Test Data")
    inspect_pickle_contents(adv_test_data, "Adversarial Test Data")
    
    # Analyze each dataset
    analyze_characteristics(train_data, "Training Set")
    analyze_characteristics(test_data, "Test Set")
    analyze_characteristics(adv_test_data, "Adversarial Test Set")
    
    # Create plots for valid data
    plot_distributions(train_data, test_data, adv_test_data)
    
    # Print comparison between normal and adversarial test sets
    print("\n=== Comparison between Normal and Adversarial Test Sets (Non-NaN values only) ===")
    for key in sorted(test_data.keys()):
        test_vals = np.array(test_data[key])[~np.isnan(test_data[key])]
        adv_vals = np.array(adv_test_data[key])[~np.isnan(adv_test_data[key])]
        
        if len(test_vals) > 0 and len(adv_vals) > 0:
            normal_mean = np.mean(test_vals)
            adv_mean = np.mean(adv_vals)
            diff_percent = ((adv_mean - normal_mean) / normal_mean) * 100
            print(f"\n{key}:")
            print(f"  Normal: mean = {normal_mean:.4f} (from {len(test_vals)} values)")
            print(f"  Adversarial: mean = {adv_mean:.4f} (from {len(adv_vals)} values)")
            print(f"  Difference: {diff_percent:+.2f}%")
        else:
            print(f"\n{key}: Insufficient valid data for comparison")

if __name__ == "__main__":
    main()
