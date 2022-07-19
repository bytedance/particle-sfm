// Copyright (C) 2014 The Regents of the University of California (Regents).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)

#include "global/orientation_util.h"

#include <Eigen/Core>
#include <ceres/rotation.h>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <colmap/base/pose.h>
#include <theia/math/graph/minimum_spanning_tree.h>

#include <theia/util/hash.h>
#include <theia/util/map_util.h>

namespace colmap {
namespace {
typedef std::pair<ImagePair, std::pair<image_t, image_t>> HeapElement;

bool SortHeapElement(const HeapElement& h1, const HeapElement& h2) {
  return h1.first.num_correspondences > h2.first.num_correspondences;
}

// Computes the orientation of the neighbor camera based on the orientation of
// the source camera and the relative rotation between the cameras.
Eigen::Vector4d ComputeOrientation(const Eigen::Vector4d& source_orientation,
                                   const ImagePair& pair,
                                   const image_t source_view_id,
                                   const image_t neighbor_view_id) {
  assert(source_view_id == pair.image_id1 || source_view_id == pair.image_id2);
  assert(neighbor_view_id == pair.image_id1 || neighbor_view_id == pair.image_id2);

  Eigen::Matrix3d source_rotation_mat = QuaternionToRotationMatrix(source_orientation);
  Eigen::Matrix3d relative_rotation = QuaternionToRotationMatrix(pair.qvec);

  const Eigen::Matrix3d neighbor_orientation =
      (source_view_id == pair.image_id1)
          ? (relative_rotation * source_rotation_mat).eval()
          : (relative_rotation.transpose() * source_rotation_mat).eval();

  Eigen::Vector4d orientation = RotationMatrixToQuaternion(neighbor_orientation);
  return orientation;
}

// Adds all the edges of view_id to the heap. Only edges that do not already
// have an orientation estimation are added.
void AddEdgesToHeap(
    const std::unordered_map<image_t, std::vector<ImagePair>>& edges,
    const std::unordered_map<image_t, Eigen::Vector4d>& orientations,
    const image_t view_id,
    std::vector<HeapElement>* heap) {
  for (const auto& edge : edges.at(view_id)) {
    // Only add edges to the heap that contain a vertex that has not been seen.
    image_t edge_id = (edge.image_id1 == view_id) ? edge.image_id2 : edge.image_id1;
    if (theia::ContainsKey(orientations, edge_id)) {
      continue;
    }

    heap->emplace_back(edge, std::make_pair(view_id, edge_id));
    std::push_heap(heap->begin(), heap->end(), SortHeapElement);
  }
}

}  // namespace

bool OrientationsFromMaximumSpanningTree(
    const std::vector<ImagePair>& view_pairs,
    std::unordered_map<image_t, Eigen::Vector4d>* orientations) {
  CHECK_NOTNULL(orientations);

  // Compute maximum spanning tree.
  theia::MinimumSpanningTree<image_t, int> mst_extractor;
  for (const auto& edge : view_pairs) {
    // Since we want the *maximum* spanning tree, we negate all of the edge
    // weights in the *minimum* spanning tree extractor.
    mst_extractor.AddEdge(
        edge.image_id1, edge.image_id2, -edge.num_correspondences);
  }

  std::unordered_set<std::pair<image_t, image_t>> mst;
  if (!mst_extractor.Extract(&mst)) {
    VLOG(2)
        << "Could not extract the maximum spanning tree from the view graph";
    return false;
  }

  // Build the neighbor edges
  std::unordered_map<std::pair<image_t, image_t>, ImagePair> pairs_map;
  for (const auto& pair : view_pairs) {
    std::pair<image_t, image_t> p{pair.image_id1, pair.image_id2};
    if (p.first > p.second) {
      std::swap(p.first, p.second);
    }

    pairs_map[p] = pair;
  }

  std::unordered_map<image_t, std::vector<ImagePair>> edges;
  for (const auto& pair : mst) {
    auto p = pair;
    if (p.first > p.second) {
      std::swap(p.first, p.second);
    }
    const auto& edge = pairs_map.at(p);
    edges[p.first].push_back(edge);
    edges[p.second].push_back(edge);
  }

  // Chain the relative rotations together to compute orientations.  We use a
  // heap to determine the next edges to add to the minimum spanning tree.
  std::vector<HeapElement> heap;

  // Set the root value.
  const image_t root_view_id = mst.begin()->first;
  (*orientations)[root_view_id] = Eigen::Vector4d(1, 0, 0, 0);
  AddEdgesToHeap(edges, *orientations, root_view_id, &heap);

  while (!heap.empty()) {
    const HeapElement next_edge = heap.front();
    // Remove the best edge.
    std::pop_heap(heap.begin(), heap.end(), SortHeapElement);
    heap.pop_back();

    // If the edge contains two vertices that have already been added then do
    // nothing.
    if (theia::ContainsKey(*orientations, next_edge.second.second)) {
      continue;
    }

    // Compute the orientation for the vertex.
    (*orientations)[next_edge.second.second] =
        ComputeOrientation(theia::FindOrDie(*orientations, next_edge.second.first),
                           next_edge.first,
                           next_edge.second.first,
                           next_edge.second.second);

    // Add all edges to the heap.
    AddEdgesToHeap(
        edges, *orientations, next_edge.second.second, &heap);
  }
  return true;
}

}
