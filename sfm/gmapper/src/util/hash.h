#ifndef COLMAP_SRC_UTIL_HASH_H_
#define COLMAP_SRC_UTIL_HASH_H_

#include <Eigen/Core>
#include <utility>
#include <theia/util/hash.h>

// This file defines hash functions for stl containers.
namespace std {
template <typename T1, typename T2, typename T3> struct hash<std::tuple<T1, T2, T3> > {
 public:
  size_t operator()(const std::tuple<T1, T2, T3>& t) const {
    size_t seed = 0;
    HashCombine(std::get<0>(t), &seed);
    HashCombine(std::get<1>(t), &seed);
    HashCombine(std::get<2>(t), &seed);
    return seed;
  }
};

}

#endif
