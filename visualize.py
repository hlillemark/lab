import glob
import os
import os.path as osp
import argparse
import random
import numpy as np
import math
from moviepy.editor import ImageSequenceClip


def save_video_grid(video, fname=None, nrow=None, fps=10):
    b, t, h, w, c = video.shape

    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    video_grid = np.zeros((t, (padding + h) * ncol + padding,
                           (padding + w) * nrow + padding, c), dtype='uint8')
    for i in range(b):
        r = i // nrow
        c = i % nrow

        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r:start_r + h, start_c:start_c + w] = video[i]


    if fname is not None:
        clip = ImageSequenceClip(list(video_grid), fps=fps)
        clip.write_gif(fname, fps=fps)
        print('saved videos to', fname)

    return video_grid # THWC, uint8

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', type=str, required=True)
args = parser.parse_args()

files = glob.glob(osp.join(args.data_path, '**', '*.npz'), recursive=True)
random.shuffle(files)

files = files[:64]
videos = []
for f in files:
    video = np.load(f)['video']
    videos.append(video)
videos = np.stack(videos, axis=0)
print(videos.shape)
save_video_grid(videos, fname='viz.gif', fps=10)
