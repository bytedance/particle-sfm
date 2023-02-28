#!/bin/bash

# set your customized python executable
# PYTHON_EXECUTABLE=/media/shaoliu/anaconda/envs/particlesfm/bin/python
PYTHON_EXECUTABLE=/data/anaconda3/envs/microtel/bin/python


# build point trajectory optimizer
cd point_trajectory/optimize
mkdir -p build && cd build
cmake -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} ..
make -j
cd ../../../

# build global mapper
cd sfm/gmapper
mkdir -p build && cd build
cmake ..
make -j
cd ../../../

