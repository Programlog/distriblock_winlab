import os
import pandas as pd
from pathlib import Path

def find_flac_files(base_dir):
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
    return pd.DataFrame(data)

# Base directory for LibriSpeech
base_dir = '/root/data_set/LibriSpeech/test-clean'

# Generate DataFrame with available files
df = find_flac_files(base_dir)

# Split the data into train/val/test sets (80/10/10 split)
train_df = df.sample(frac=0.8, random_state=42)
remaining = df.drop(train_df.index)
val_df = remaining.sample(frac=0.5, random_state=42)
test_df = remaining.drop(val_df.index)

# Save the new CSVs
output_dir = 'results/transformer/74443'
train_df.to_csv(f'{output_dir}/src_train_trans.csv', index=False)
val_df.to_csv(f'{output_dir}/src_val_trans.csv', index=False)
test_df.to_csv(f'{output_dir}/src_test_trans.csv', index=False)

# Also create adversarial versions (you can modify this as needed)
train_df.head(100).to_csv(f'{output_dir}/adv_train_trans.csv', index=False)
test_df.head(100).to_csv(f'{output_dir}/adv_test_trans.csv', index=False)

print(f"Created new CSV files with {len(df)} total files:")
print(f"Train: {len(train_df)} files")
print(f"Val: {len(val_df)} files")
print(f"Test: {len(test_df)} files")

