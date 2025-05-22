# This script computes and plots a histogram of log(depth) values from the first 100 .npz files in lab/teco_data_training.
# It skips files where all depth values are zero or negative.
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# Script to compute and plot histogram of log(depth) values from first 100 .npz files in lab/teco_data_training

def find_npz_files(root, max_files=100):
    npz_files = []
    for dirpath, _, _ in os.walk(root):
        npz_files.extend(sorted(glob.glob(os.path.join(dirpath, '*.npz'))))
        if len(npz_files) >= max_files:
            break
    return npz_files[:max_files]

npz_files = find_npz_files('./teco_data_training', max_files=100)

all_log_depths = []

for f in npz_files:
    data = np.load(f)
    depth = data['depth_video']
    # Flatten and filter out non-positive values
    depth_flat = depth.flatten()
    depth_flat = depth_flat[depth_flat > 0]
    if depth_flat.size == 0:
        print(f"WARNING: {f} has no positive depth values, skipping.")
        continue
    log_depth = np.log(depth_flat)
    all_log_depths.append(log_depth)
    print(f"{f}: log(depth) min={log_depth.min()}, max={log_depth.max()}")

if all_log_depths:
    all_log_depths = np.concatenate(all_log_depths)
    print(f"\nGlobal log(depth) min: {all_log_depths.min()}")
    print(f"Global log(depth) max: {all_log_depths.max()}")

    plt.hist(all_log_depths, bins=50)
    plt.title('Histogram of log(Depth) Values (first 100 .npz files)')
    plt.xlabel('log(Depth Value)')
    plt.ylabel('Frequency')
    plt.savefig('log_depth_histogram.png')
    print('Log-depth histogram saved to log_depth_histogram.png')
else:
    print("No valid log(depth) values found in the selected files.") 