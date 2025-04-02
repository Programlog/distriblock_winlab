import os
import pandas as pd
from pathlib import Path

def find_flac_files(base_dir, max_files=100):  # Added max_files parameter
    """Find all .flac files in the directory and create CSV entries"""
    data = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.flac'):
                full_path = os.path.join(root, file)
                # Extract speaker ID from the filename (assuming format like 1089-134691-0025.flac)
                filename = os.path.basename(full_path)
                file_id = filename.rsplit('.', 1)[0]  # Remove .flac extension
                spk_id = '-'.join(file_id.split('-')[:2])  # Get speaker ID part
                
                data.append({
                    'ID': file_id,
                    'duration': 0,  # This will be placeholder
                    'wav': full_path,
                    'spk_id': spk_id,
                    'wrd': 'PLACEHOLDER'  # This will be placeholder
                })
                
                # Break if we've collected enough files
                if len(data) >= max_files:
                    return pd.DataFrame(data)
                    
    return pd.DataFrame(data)

# Base directory for LibriSpeech
base_dir = '/root/data_set/LibriSpeech/test-clean'

# Define smaller dataset sizes
TOTAL_FILES = 50  # Total number of files to use
TRAIN_SIZE = 30    # 60% for training
VAL_SIZE = 10      # 20% for validation
TEST_SIZE = 10     # 20% for testing

# Generate DataFrame with limited files
df = find_flac_files(base_dir, max_files=TOTAL_FILES)

# Split the data into train/val/test sets
train_df = df.sample(n=TRAIN_SIZE, random_state=42)
remaining = df.drop(train_df.index)
val_df = remaining.sample(n=VAL_SIZE, random_state=42)
test_df = remaining.drop(val_df.index)

# Save the new CSVs
output_dir = 'results/transformer/74443'
train_df.to_csv(f'{output_dir}/src_train_trans.csv', index=False)
val_df.to_csv(f'{output_dir}/src_val_trans.csv', index=False)
test_df.to_csv(f'{output_dir}/src_test_trans.csv', index=False)

# Create adversarial versions with same size as test set
adv_train_df = train_df.head(TEST_SIZE)  # Take first TEST_SIZE samples
adv_test_df = test_df.copy()  # Use all test samples

adv_train_df.to_csv(f'{output_dir}/adv_train_trans.csv', index=False)
adv_test_df.to_csv(f'{output_dir}/adv_test_trans.csv', index=False)

print(f"Created new CSV files with {len(df)} total files:")
print(f"Train: {len(train_df)} files")
print(f"Val: {len(val_df)} files")
print(f"Test: {len(test_df)} files")
print(f"Adv Train: {len(adv_train_df)} files")
print(f"Adv Test: {len(adv_test_df)} files")
