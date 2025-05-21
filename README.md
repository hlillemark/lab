# DMLab

A fork of the [Teco project's dmlab](https://github.com/wilson1yan/lab) which is a fork of the [original DMLab repo](https://github.com/deepmind/lab)


# Installation

To install it as a python package in a conda environment, follow these instructions. If you don't have sudo access, then remove the sudo commands and try with docker. If you already have bazel installed, can remove the lines related to installing those.:
```
conda create -n dmlab-data python=3.10
conda activate dmlab-data
pip install -r requirements.txt
bash ./install-dmlab.sh
```

Based on the script found [here](https://gist.github.com/danijar/ca6ab917188d2e081a8253b3ca5c36d3), with a few modifications



# Collecting

Collecting using python to run, check the args yourself (e.g. num parallel, num trajectories, height, width, etc.):

`python parallel_random_agent_traverse.py`

You may need to run these two lines if it errors like `version 'GLIBCXX_3.4.30' not found (required by /lib/x86_64-linux-gnu/libLLVM-15.so.1)`:
```
unset LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

\[Version from original repo\]: if you can figure out how to get bazel to work with the install:

`python collect.py -d data_path`


# Adding a new map

The original lab version uses lua files to do procedural map generation. Due to the bazel build not working well with python in its current form, you may first create a new map, such as one based off `lab/game_scripts/levels/demos/random_maze.lua`, and then try passing that into the lab generation script as the level. If it does not recognize it, as a workaround, just copy that file into the location it outputs in the error message, which is likely your conda path to the built package. Likely, rebuilding the python package would work better, but I have not tried it yet. 

