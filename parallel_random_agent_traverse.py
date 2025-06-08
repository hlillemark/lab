import subprocess
from pathlib import Path
from multiprocessing import Process
import argparse
import os

# Unset LD_LIBRARY_PATH if it's present in the environment
if 'LD_LIBRARY_PATH' in os.environ:
    del os.environ['LD_LIBRARY_PATH']

# Set LD_PRELOAD, which will be inherited by all child processes
os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libstdc++.so.6'


def run_process(proc_id, n_traj, height, width, length, base_output_dir, level_script):
    output_dir = Path(base_output_dir) / str(proc_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3", "./python/random_agent_traverse.py",
        "--output_dir", str(output_dir),
        "--n_traj", str(n_traj),
        "--height", str(height),
        "--width", str(width),
        "--length", str(length),
        "--level_script", level_script,
    ]
    subprocess.run(cmd)

def main(n_parallel, total_n_traj, height, width, length, output_dir, level_script):
    base_traj = total_n_traj // n_parallel
    remainder = total_n_traj % n_parallel

    processes = []
    for i in range(n_parallel):
        # Distribute the remainder across the first few processes
        n_traj_i = base_traj + (1 if i < remainder else 0)
        if n_traj_i == 0:
            continue
        p = Process(target=run_process, args=(i + 1, n_traj_i, height, width, length, output_dir, level_script))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Count and check size of npz files
    total_size = 0
    num_files = 0
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.npz'):
                num_files += 1
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
    
    print(f"Found {num_files} .npz files")
    print(f"Total size: {total_size / (1024*1024*1024):.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_parallel", type=int, default=32)
    parser.add_argument("--n_traj", type=int, default=512)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="/data/hansen/projects/bh/wm-memory/data/dmlab/cal_stats")
    parser.add_argument("--level_script", type=str, default="demos/random_maze")
    args = parser.parse_args()

    main(
        n_parallel=args.n_parallel,
        total_n_traj=args.n_traj,
        height=args.height,
        width=args.width,
        length=args.length,
        output_dir=args.output_dir,
        level_script=args.level_script
    )
