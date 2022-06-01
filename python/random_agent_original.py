# Copyright 2016 Google Inc.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""Basic random agent for DeepMind Lab."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import argparse
import random
import numpy as np
import six
import skvideo.io
from PIL import Image

import deepmind_lab


def _action(*entries):
  return np.array(entries, dtype=np.intc)


class DiscretizedRandomAgent(object):
  """Simple agent for DeepMind Lab."""

  ACTIONS = {
      'look_left': _action(-20, 0, 0, 0, 0, 0, 0),
      'look_right': _action(20, 0, 0, 0, 0, 0, 0),
      'look_up': _action(0, 10, 0, 0, 0, 0, 0),
      'look_down': _action(0, -10, 0, 0, 0, 0, 0),
      'strafe_left': _action(0, 0, -1, 0, 0, 0, 0),
      'strafe_right': _action(0, 0, 1, 0, 0, 0, 0),
      'forward': _action(0, 0, 0, 1, 0, 0, 0),
      'backward': _action(0, 0, 0, -1, 0, 0, 0),
      'fire': _action(0, 0, 0, 0, 1, 0, 0),
      'jump': _action(0, 0, 0, 0, 0, 1, 0),
      'crouch': _action(0, 0, 0, 0, 0, 0, 1)
  }

  ACTION_LIST = list(six.viewvalues(ACTIONS))

  def step(self, unused_reward, unused_image):
    """Gets an image state and a reward, returns an action."""
    return random.choice(DiscretizedRandomAgent.ACTION_LIST)


def run(length, width, height, fps, level):
  """Spins up an environment and runs the random agent."""
  config = {
      'fps': str(fps),
      'width': str(width),
      'height': str(height)
  }
  env = deepmind_lab.Lab(level, ['RGB_INTERLEAVED', 'DEPTH', 'PROJECTION_MATRIX', 
                                 'MODELVIEW_MATRIX', 'DEBUG.POS.TRANS', 'DEBUG.POS.ROT'], config=config)

  env.reset()

  rgb_frames, depth_frames = [], []
  pms, mvms = [], []
  eyes, angles = [], []
  for i in six.moves.range(length + 15):
    if not env.is_running():
      print('Environment stopped early')
      env.reset()
    obs = env.observations()

    if i >= 15:
        pms.append(obs['PROJECTION_MATRIX'])
        mvms.append(obs['MODELVIEW_MATRIX'])
        rgb_frames.append(obs['RGB_INTERLEAVED'])
        depth_frames.append(obs['DEPTH'])
        eyes.append(obs['DEBUG.POS.TRANS'].astype(np.float32))
        angles.append(obs['DEBUG.POS.ROT'].astype(np.float32))

    env.step(DiscretizedRandomAgent.ACTIONS['look_left'], num_steps=4)

  pms = np.stack(pms)
  mvms = np.stack(mvms)
  rgb_frames = np.stack(rgb_frames)
  depth_frames = np.stack(depth_frames)
  eyes = np.stack(eyes)
  angles = np.stack(angles)

  os.makedirs(args.output, exist_ok=True)
  skvideo.io.vwrite(osp.join(args.output, 'video.mp4'), rgb_frames)

  np.savez_compressed(osp.join(args.output, 'data.npz'), rgb=rgb_frames, depth=depth_frames, 
                      projection_matrix=pms, modelview_matrix=mvms, eye=eyes, angle=angles)
  first_frame = rgb_frames[0]
  first_frame = Image.fromarray(first_frame)
  first_frame.save(osp.join(args.output, 'first_frame.png'))

  first_frame_depth = depth_frames[0].repeat(3, axis=-1)
  first_frame_depth = (first_frame_depth - first_frame_depth.min()) / (first_frame_depth.max() - first_frame_depth.min())
  first_frame_depth = Image.fromarray((first_frame_depth * 255).astype(np.uint8))
  first_frame_depth.save(osp.join(args.output, 'first_frame_depth.png'))



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--length', type=int, default=30,
                      help='Number of steps to run the agent')
  parser.add_argument('--width', type=int, default=128,
                      help='Horizontal size of the observations')
  parser.add_argument('--height', type=int, default=128,
                      help='Vertical size of the observations')
  parser.add_argument('--fps', type=int, default=30,
                      help='Number of frames per second')
  parser.add_argument('--runfiles_path', type=str, default=None,
                      help='Set the runfiles path to find DeepMind Lab data')
  parser.add_argument('--level_script', type=str,
                      default='demos/random_maze',
                      help='The environment level script to load')
  parser.add_argument('--output', type=str, default='/home/wilson/repos/lab/trajectory')

  args = parser.parse_args()
  if args.runfiles_path:
    deepmind_lab.set_runfiles_path(args.runfiles_path)
  run(args.length, args.width, args.height, args.fps, args.level_script)
