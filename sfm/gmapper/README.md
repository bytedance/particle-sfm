# gcolmap
Global structure-from-motion with [COLMAP](https://colmap.github.io/) database. Multiple features from [TheiaSfM](http://theia-sfm.org/) were adapted to fit the COLMAP database format, including L1-IRLS rotation averaging, known-rotation translation refinement, LUD translation averaging, etc.

# installation 
* COLMAP [[Guide](https://colmap.github.io/install.html)]
* Theia SfM (customized version) [[Guide](../../misc/doc/theia.md)]

To compile gcolmap:
```
mkdir build && cd build
cmake ..
make -j
cd ..
```

