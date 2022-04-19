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
import itertools
import os.path as osp
import argparse
import random
import numpy as np
from tqdm import tqdm
import six
from PIL import Image

import deepmind_lab


def _action(*entries):
  return np.array(entries, dtype=np.intc)


class GoalAgent(object):
    ACTIONS = {
        'look_left': _action(-20, 0, 0, 0, 0, 0, 0),
        'look_right': _action(20, 0, 0, 0, 0, 0, 0),
        'forward': _action(0, 0, 0, 1, 0, 0, 0),
        'look_left_forward': _action(-20, 0, 0, 1, 0, 0, 0),
        'look_right_forward': _action(20, 0, 0, 1, 0, 0, 0)
    }

    ACTIONS_TO_IDX = {
        'look_left': 0,
        'look_right': 1,
        'forward': 2,
        'look_left_forward': 3,
        'look_right_forward': 4
    }

    ACTION_LIST = list(six.viewvalues(ACTIONS))

    # yaw
    # 90: north
    # 0: east
    # -90: south
    # 180/-180: west

    def _get_rotation(self, rot, target):
        ranges_for_right, ranges_for_left = [], []
        d = 0
        if target > 0:
            ranges_for_right.append((target, 180))
            leftover = target
            ranges_for_right.append((-180, -180 + leftover))

            ranges_for_left.append((target - 180, target))


        if target <= 0:
            ranges_for_right.append((target, target + 180))
            
            ranges_for_left.append((-180, target))
            ranges_for_left.append((180 - abs(target), 180))

        assert sum([r - l for l, r in ranges_for_right]) == 180
        assert sum([r - l for l, r in ranges_for_left]) == 180

        for l, r in ranges_for_right:
            if l <= rot <= r:
                return 'look_right'

        for l, r in ranges_for_left:
            if l <= rot <= r:
                return 'look_left'

        raise Exception()

    def move_to_target(self, pos, rot, target):
        dx, dy = [p2 - p1 for p1, p2 in zip(pos, target)]
        if dx == 0:
            angle = 90 if dy >= 0 else -90
        else:
            angle = np.arctan(abs(dy / dx)) * 180 / np.pi

        if dx >= 0 and dy >= 0:
            angle = angle
        elif dx >= 0 and dy <= 0:
            angle = -angle
        elif dx <= 0 and dy >= 0:
            angle = 180 - angle
        else:
            angle = -180 + angle

        turn_dir = self._get_rotation(rot[1], angle)
        if abs(rot[1] - angle) < 10 or abs(rot[1] - 360 - angle) < 10 or abs(rot[1] + 360 - angle) < 10:
            action = f'{turn_dir}_forward'
        else:
            action = turn_dir
        return action


    def step(self, timestep, pos, rot, target):
        if timestep < 40:
            return self.ACTIONS['look_left'], 0

        action = self.move_to_target(pos, rot, target)
        return self.ACTIONS[action], self.ACTIONS_TO_IDX[action]


class Maze:
    WALL_SYMBOL = '*'
    MAZE_CELL_SIZE = 100

    def __init__(self, maze_str):
        maze_str = maze_str.strip()
        lines = maze_str.split('\n')

        height = len(lines)
        width = 0
        for line in lines:
            width = max(width, len(line))

        maze = np.zeros((width, height), dtype=np.int32)

        for j, line in enumerate(lines):
            for i, cell in enumerate(line):
                if cell == self.WALL_SYMBOL:
                    maze[i, j] = 1
        self.maze = maze

    def to_world_coord(self, x, y):
        maze = self.maze
        y = maze.shape[1] - y - 1
        return (float(x) + 0.5) * self.MAZE_CELL_SIZE, (float(y) + 0.5) * self.MAZE_CELL_SIZE

    def to_maze_coord(self, x, y):
        maze = self.maze
        x = int(x / self.MAZE_CELL_SIZE)
        y = int(y / self.MAZE_CELL_SIZE)
        y = maze.shape[1] - y - 1
        return x, y

    def _get_neighbors(self, x, y):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if (dx == 0 and dy == 0) or abs(dx * dy) == 1:
                    continue
                xn, yn = x + dx, y + dy
                if xn < 0 or xn >= self.maze.shape[0]:
                    continue
                if yn < 0 or yn >= self.maze.shape[1]:
                    continue
                if self.maze[xn, yn] == 0:
                    neighbors.append((xn, yn))
        return neighbors

    def sample_goal_path(self, x, y, dist):
        maze = self.maze
        x, y = self.to_maze_coord(x, y)
        assert maze[x, y] == 0, 'x-y pos must not be a wall'
        assert dist > 0

        dist_to_cur = np.zeros_like(maze)
        def fill_bfs(root):
            queue = [root]
            while queue:
                node = queue.pop(0)
                neighbors = self._get_neighbors(*node)
                for n in neighbors:
                    if dist_to_cur[n] == 0 and n != root:
                        dist_to_cur[n] = dist_to_cur[node] + 1
                        queue.append(n)
        fill_bfs((x, y))
        xs, ys = np.where(dist_to_cur == dist)
        assert len(xs) > 0
        idx = np.random.randint(0, len(xs))
        xg, yg = xs[idx], ys[idx]

        landmarks = []
        xc, yc = xg, yg
        while (xc, yc) != (x, y):
            ns = self._get_neighbors(xc, yc)
            n = min(ns, key=lambda n: dist_to_cur[n])
            landmarks.insert(0, n)
            xc, yc = n
        landmarks = landmarks[1:] + [(xg, yg)] + landmarks[::-1]
        landmarks = [self.to_world_coord(*lm) for lm in landmarks]
        return np.array(landmarks)


def sample_trajectory(env, agent, length, name, goal_dist, skip=10):
  env.reset()
  frames = []
  #top_down = []
  actions = []
  
  obs = env.observations()
  maze = Maze(obs['DEBUG.MAZE.LAYOUT'])
  pos = obs['DEBUG.POS.TRANS']
  rot = obs['DEBUG.POS.ROT']

  cur_idx = 0
  path = maze.sample_goal_path(pos[0], pos[1], goal_dist)

  for t in range(length + skip):
    obs = env.observations()
    pos = obs['DEBUG.POS.TRANS']
    rot = obs['DEBUG.POS.ROT']
    if np.linalg.norm(pos[:2] - path[cur_idx]) <= 10:
        cur_idx = (cur_idx + 1) % len(path)

    action, idx = agent.step(t, pos, rot, path[cur_idx])

    if t >= skip:
        frames.append(obs['RGB_INTERLEAVED'].copy())
        #top_down.append(obs['DEBUG.CAMERA_INTERLEAVED.TOP_DOWN'].copy())
        actions.append(idx)
    env.step(action, num_steps=4)

  video = np.stack(frames, axis=0)
  #top_down = np.stack(top_down, axis=0)
  actions = np.array(actions).astype(int)
  filepath = osp.join(args.output_dir, f'{name}.npz')
  np.savez(filepath, video=video, actions=actions)


def sample_trajectories(n, env, agent, length, goal_dist):
    i = 0
    pbar = tqdm(total=n)
    while i < n:
        success = sample_trajectory(env, agent, length, i, goal_dist)
        pbar.update(1)
        i += 1
    pbar.close()


def run(length, width, height, fps, level, record, demo, demofiles, video):
  """Spins up an environment and runs the random agent."""
  config = {
      'fps': str(fps),
      'width': str(width),
      'height': str(height)
  }
  if record:
    config['record'] = record
  if demo:
    config['demo'] = demo
  if demofiles:
    config['demofiles'] = demofiles
  if video:
    config['video'] = video

  os.makedirs(args.output_dir, exist_ok=True)

  # DEBUG.CAMERA_INTERLEAVED.TOP_DOWN
  env = deepmind_lab.Lab(level, ['RGB_INTERLEAVED', 'DEBUG.POS.TRANS', 'DEBUG.POS.ROT', 'DEBUG.MAZE.LAYOUT'], config=config)
  agent = GoalAgent()
  sample_trajectories(args.n_traj, env, agent, length, args.goal_dist)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--length', type=int, default=100,
                      help='Number of steps to run the agent')
  parser.add_argument('--width', type=int, default=64,
                      help='Horizontal size of the observations')
  parser.add_argument('--height', type=int, default=64,
                      help='Vertical size of the observations')
  parser.add_argument('--goal_dist', type=int, default=1,
                      help='Sample goal # steps away')
  parser.add_argument('--fps', type=int, default=30,
                      help='Number of frames per second')
  parser.add_argument('--runfiles_path', type=str, default=None,
                      help='Set the runfiles path to find DeepMind Lab data')
  parser.add_argument('--level_script', type=str,
                      default='demos/random_maze',
                      help='The environment level script to load')
  parser.add_argument('--n_traj', type=int, default=64)
  parser.add_argument('--output_dir', type=str, default='/shared/wilson/datasets/dl_maze')


  parser.add_argument('--record', type=str, default=None,
                      help='Record the run to a demo file')
  parser.add_argument('--demo', type=str, default=None,
                      help='Play back a recorded demo file')
  parser.add_argument('--demofiles', type=str, default=None,
                      help='Directory for demo files')
  parser.add_argument('--video', type=str, default=None,
                      help='Record the demo run as a video')

  args = parser.parse_args()
  if args.runfiles_path:
    deepmind_lab.set_runfiles_path(args.runfiles_path)
  run(args.length, args.width, args.height, args.fps, args.level_script,
      args.record, args.demo, args.demofiles, args.video)
