import os
import os.path as osp
import argparse
import multiprocessing as mp


def worker(i, args):
    output_dir = osp.join(args.data_path, f'{i}')
    n_traj = args.n_traj // args.n_parallel + (i < (args.n_traj % args.n_parallel))
    cmd = f'bazel run :python_random_agent_traverse -- --length {args.length} --width {args.width} --height {args.height} --fps {args.fps} --output_dir {output_dir} --n_traj {n_traj}'
    os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, required=True)
    parser.add_argument('-n', '--n_traj', type=int, default=40000)
    parser.add_argument('-l', '--length', type=int, default=100)
    parser.add_argument('--width', type=int, default=64,
                        help='Horizontal size of the observations')
    parser.add_argument('--height', type=int, default=64,
                        help='Vertical size of the observations')
    parser.add_argument('--fps', type=int, default=30,
                        help='Number of frames per second')
    parser.add_argument('-p', '--n_parallel', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.data_path, exist_ok=True)

    procs = [mp.Process(target=worker, args=(i, args)) for i in range(args.n_parallel)]
    [p.start() for p in procs]
    [p.join() for p in procs]

