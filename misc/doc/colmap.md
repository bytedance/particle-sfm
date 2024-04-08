# Colmap installation

First install packages to support Colmap:
```
sudo apt-get install \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev

Clone colmap, checkout the version which works with this repo, compile and install:
```
git clone https://github.com/colmap/colmap
git checkout bd84ad6

cd colmap
mkdir build
cd build
cmake .. -GNinja
ninja
sudo ninja install
```