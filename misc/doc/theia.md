# Theia installation
First install packages to support Theia:
```
sudo apt-get install -y libopenimageio-dev librocksdb-dev rapidjson-dev freeglut3-dev
```
Clone the customized repo, compile and install:
```
git clone https://github.com/B1ueber2y/TheiaSfM
cd TheiaSfM
git checkout upstream/particle-sfm
mkdir build && cd build
cmake ..
make -j
sudo make install
```
