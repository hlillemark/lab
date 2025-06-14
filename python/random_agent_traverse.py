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
from scipy.spatial.transform import Rotation

import deepmind_lab
from utils import getRawDepth

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
            action = 'forward'
        else:
            action = turn_dir
        return action


    def step(self, timestep, pos, rot, target):
        if timestep < 30:
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

        self.width, self.height = width, height
        self.reset_visited()


    def reset_visited(self):
        self.visited = np.zeros((self.width, self.height), dtype=bool)
        for i in range(self.width):
            for j in range(self.height):
                if self.maze[i, j] == 1:
                    self.visited[i, j] = True

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

    def sample_goal_path(self, x, y):
        maze = self.maze
        x_o, y_o = x, y
        x, y = self.to_maze_coord(x, y)
        assert maze[x, y] == 0, 'x-y pos must not be a wall'

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

        max_dist = (~self.visited * dist_to_cur).max()
        if max_dist == 0:
            self.reset_visited()
            max_dist = (~self.visited * dist_to_cur).max()

        xs, ys = np.where(dist_to_cur == max_dist)
        assert len(xs) > 0
        idx = np.random.randint(0, len(xs))
        xg, yg = xs[idx], ys[idx]

        self.visited[x, y] = True
        landmarks = []
        xc, yc = xg, yg
        while (xc, yc) != (x, y):
            self.visited[xc, yc] = True
            ns = self._get_neighbors(xc, yc)
            n = min(ns, key=lambda n: dist_to_cur[n])
            landmarks.insert(0, n)
            xc, yc = n
        landmarks = landmarks[1:] + [(xg, yg)]
        landmarks = [self.to_world_coord(*lm) for lm in landmarks]
        return np.array(landmarks)


def sample_trajectory(env, agent, length, name, skip=10):
    env.reset()
    frames = []
    depth_frames = []
    proj_matrices = []
    mv_matrices = []
    poss = []
    rots = []
    actions = []
    camera_poss = []
    camera_rots = []
	
    obs = env.observations()
    maze = Maze(obs['DEBUG.MAZE.LAYOUT'])
    pos = obs['DEBUG.POS.TRANS']
    rot = obs['DEBUG.POS.ROT']
    
    # Camera position and rotation 
    camera_pos = obs['DEBUG.PLAYERS.EYE.POS']
    camera_rot = obs['DEBUG.PLAYERS.EYE.ROT']

    cur_idx = 0
    path = maze.sample_goal_path(pos[0], pos[1])

    for t in range(length + skip):
        obs = env.observations()
        pos = obs['DEBUG.POS.TRANS']
        rot = obs['DEBUG.POS.ROT']
        
        # Camera position and rotation 
        camera_pos = obs['DEBUG.PLAYERS.EYE.POS']
        camera_rot = obs['DEBUG.PLAYERS.EYE.ROT']
        
        if np.linalg.norm(pos[:2] - path[cur_idx]) <= 40:
            cur_idx += 1
            if cur_idx >= len(path):
                path = maze.sample_goal_path(pos[0], pos[1])
                cur_idx = 0

        action, idx = agent.step(t, pos, rot, path[cur_idx])

        if t >= skip:
            frames.append(obs['RGB_INTERLEAVED'].copy())
            actions.append(idx)
            depth_frames.append(getRawDepth(obs['DEPTH'].copy(), obs['PROJECTION_MATRIX'].copy()))
            proj_matrices.append(obs['PROJECTION_MATRIX'].copy())

            mv = obs['MODELVIEW_MATRIX'].copy()
            mv = np.linalg.inv(mv)
            rot, pos = mv[:3, :3], mv[:3, -1]
            rot = Rotation.from_matrix(rot).as_quat().astype(np.float32)
            if np.any(np.isnan(rot)):
                print('ERROR', mv, rot)
			
            mv_matrices.append(mv)
            poss.append(pos)
            rots.append(rot)
            camera_poss.append(camera_pos)
            camera_rots.append(camera_rot)
        env.step(action, num_steps=4)

    video = np.stack(frames, axis=0)
    actions = np.array(actions).astype(int)
    depth_frames = np.array(depth_frames).astype(np.float32)
    proj_matrices = np.array(proj_matrices).astype(np.float32)
    mv_matrices = np.array(mv_matrices).astype(np.float32)
    poss = np.array(poss).astype(np.float32)
    rots = np.array(rots).astype(np.float32)
    camera_poss = np.array(camera_poss).astype(np.float32)
    camera_rots = np.array(camera_rots).astype(np.float32)
    
    filepath = osp.join(args.output_dir, f'{name}.npz')
    if args.rgb_only:
        np.savez(filepath, video=video, actions=actions)
    else:
        np.savez(filepath, video=video, actions=actions,
                 depth_video=depth_frames, proj_matrices=proj_matrices,
                 mv_matrices=mv_matrices, pos=poss, rot=rots,
                 camera_pos=camera_poss, camera_rot=camera_rots)

def sample_trajectories(n, env, agent, length):
    i = 0
    pbar = tqdm(total=n)
    while i < n:
        success = sample_trajectory(env, agent, length, i)
        pbar.update(1)
        i += 1
    pbar.close()


def run(length, width, height, fps, level):
  """Spins up an environment and runs the random agent."""
  config = {
      'fps': str(fps),
      'width': str(width),
      'height': str(height)
  }

  os.makedirs(args.output_dir, exist_ok=True)

  env = deepmind_lab.Lab(level, ['RGB_INTERLEAVED', 'DEPTH', 'PROJECTION_MATRIX', 'MODELVIEW_MATRIX',
                                 'DEBUG.POS.TRANS', 'DEBUG.POS.ROT', 'DEBUG.MAZE.LAYOUT', 'DEBUG.PLAYERS.EYE.POS', 'DEBUG.PLAYERS.EYE.ROT'], config=config)
  agent = GoalAgent()
  sample_trajectories(args.n_traj, env, agent, length)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--length', type=int, default=300,
                      help='Number of steps to run the agent')
  parser.add_argument('--width', type=int, default=64,
                      help='Horizontal size of the observations')
  parser.add_argument('--height', type=int, default=64,
                      help='Vertical size of the observations')
  parser.add_argument('--fps', type=int, default=30,
                      help='Number of frames per second')
  parser.add_argument('--runfiles_path', type=str, default=None,
                      help='Set the runfiles path to find DeepMind Lab data')
  parser.add_argument('--level_script', type=str,
                      default='demos/random_maze',
                      help='The environment level script to load')
  parser.add_argument('--n_traj', type=int, default=40000)
  parser.add_argument('--output_dir', type=str, required=True)
  parser.add_argument('--rgb_only', action='store_true')


  args = parser.parse_args()
  if args.runfiles_path:
    deepmind_lab.set_runfiles_path(args.runfiles_path)
  run(args.length, args.width, args.height, args.fps, args.level_script)
