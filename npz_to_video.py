import numpy as np
import imageio.v3 as iio

data = np.load('./test_debug/0.npz')

video = data['video']

video = np.uint8(video * 255)
iio.imwrite("example.mp4", video, fps=30)  # adjust fps as needed

depth = data['depth_video']

import pdb; pdb.set_trace()
breakpoint()

depth = np.uint8(depth * 255)
iio.imwrite('example_depth.mp4', depth, fps=30)
