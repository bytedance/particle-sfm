# Ceres installation
```
git clone https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver
mkdir build && cd build
cmake -DBUILD_SHARED_LIBS=ON ..
make -j
sudo make install
```