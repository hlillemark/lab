import os
import glob
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from tqdm import tqdm
# Script to compute global min/max and plot histogram of depth values from first 100 .npz files in lab/teco_data_training

def find_npz_files(root, max_files=2000):
    npz_files = []
    for dirpath, _, _ in os.walk(root):
        npz_files.extend(sorted(glob.glob(os.path.join(dirpath, '*.npz'))))
        if len(npz_files) >= max_files:
            break
    return npz_files[:max_files]

npz_files = find_npz_files('./teco_data_training', max_files=100)

all_depths = []
global_min = float('inf')
global_max = float('-inf')

all_rgb = []

for f in tqdm(npz_files):
    data = np.load(f)
    depth = data['depth_video']
    min_val = depth.min()
    max_val = depth.max()
    global_min = min(global_min, min_val)
    global_max = max(global_max, max_val)
    all_depths.append(depth.flatten())
    print(f"{f}: min={min_val}, max={max_val}")
    rgb = data['video']
    all_rgb.append(rgb.reshape(-1, 3))

all_depths = np.concatenate(all_depths)
print(f"\nGlobal min: {global_min}")
print(f"Global max: {global_max}")

# Plot regular histogram
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.hist(all_depths, bins=50)
# plt.title('Histogram of Depth Values\n(first 100 .npz files)')
# plt.xlabel('Depth Value')
# plt.ylabel('Frequency')

# # Plot log depth histogram
# plt.subplot(1, 2, 2)
# plt.hist(np.log(all_depths + 1), bins=50)  # Add 1 to handle zeros
# plt.title('Histogram of Log Depth Values\n(first 100 .npz files)')
# plt.xlabel('Log Depth Value')
# plt.ylabel('Frequency')

# plt.tight_layout()
# plt.savefig('depth_histograms.png')
# print('Histograms saved to depth_histograms.png')
# plt.show()  # Optionally, you can still show the plot interactively


# Plot RGB histograms
plt.figure(figsize=(12, 5))

# all_rgb = []
# for f in tqdm(npz_files):
#     data = np.load(f)
#     rgb = data['rgb_video']
#     all_rgb.append(rgb.reshape(-1, 3))

all_rgb = np.concatenate(all_rgb, axis=0)

plt.subplot(1, 1, 1)
for i, color in enumerate(['r', 'g', 'b']):
    plt.hist(all_rgb[:,i], bins=50, color=color, alpha=0.5, label=color.upper())
plt.title('Histogram of RGB Values\n(first 100 .npz files)')
plt.xlabel('RGB Value')
plt.ylabel('Frequency') 
plt.legend()

# plt.subplot(1, 2, 2)
# for i, color in enumerate(['r', 'g', 'b']):
#     plt.hist(np.log(all_rgb[:,i] + 1), bins=50, color=color, alpha=0.5, label=color.upper())
# plt.title('Histogram of Log RGB Values\n(first 100 .npz files)')
# plt.xlabel('Log RGB Value')
# plt.ylabel('Frequency')
# plt.legend()

plt.tight_layout()
plt.savefig('rgb_histograms.png')
print('RGB histograms saved to rgb_histograms.png')
