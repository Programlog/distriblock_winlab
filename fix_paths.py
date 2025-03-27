import pandas as pd
import os
from pathlib import Path

# Read the CSV file
csv_path = "results/transformer/74443/src_train_trans.csv"
df = pd.read_csv(csv_path)

# Function to find the actual path of a file
def find_file(filename, search_dir):
    for root, dirs, files in os.walk(search_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

# Update paths in the DataFrame
base_dir = "/home/ubuntu/data_set/LibriSpeech"
new_paths = []
missing_files = []

for index, row in df.iterrows():
    # Extract filename from the original path
    filename = os.path.basename(row['wav'])
    
    # Find the actual path
    actual_path = find_file(filename, base_dir)
    if actual_path:
        new_paths.append(actual_path)
    else:
        missing_files.append(filename)
        new_paths.append(row['wav'])  # Keep original path if file not found

# Update the DataFrame
df['wav'] = new_paths

# Save the updated CSV
df.to_csv(csv_path + ".fixed", index=False)

# Print summary
print(f"Total files processed: {len(df)}")
print(f"Files not found: {len(missing_files)}")
if missing_files:
    print("First few missing files:")
    for f in missing_files[:5]:
        print(f"  {f}")
