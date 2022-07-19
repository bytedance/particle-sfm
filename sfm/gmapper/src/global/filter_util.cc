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

#include "global/filter_util.h"

#include <ceres/rotation.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glog/logging.h>
#include <memory>
#include <mutex>  // NOLINT
#include <unordered_map>
#include <unordered_set>

#include <colmap/base/pose.h>
#include <colmap/util/math.h>
#include <colmap/util/threading.h>
#include <theia/math/graph/connected_components.h>
#include <theia/util/map_util.h>

#include <theia/util/random.h>

namespace colmap {

using Eigen::Vector3d;
using Eigen::Vector4d;

namespace {

bool AngularDifferenceIsAcceptable(
    const Eigen::Vector4d& orientation1,
    const Eigen::Vector4d& orientation2,
    const Eigen::Vector4d& relative_orientation,
    const double sq_max_relative_rotation_difference_radians) {
  const Eigen::Vector4d composed_relative_rotation =
      ConcatenateQuaternions(InvertQuaternion(orientation1), orientation2);
  const Eigen::Vector4d loop_rotation =
      ConcatenateQuaternions(composed_relative_rotation, InvertQuaternion(relative_orientation));
  Eigen::Vector3d a;
  ceres::QuaternionToAngleAxis(loop_rotation.data(), a.data());
  const double sq_rotation_angular_difference_radians = a.squaredNorm();
  return sq_rotation_angular_difference_radians <=
         sq_max_relative_rotation_difference_radians;
}

// Helper struct to maintain the graph for the translation projection problem.
struct MFASNode {
  std::unordered_map<image_t, double> incoming_nodes;
  std::unordered_map<image_t, double> outgoing_nodes;
  double incoming_weight = 0;
  double outgoing_weight = 0;
};

// Rotate the translation direction based on the known orientation such that the
// translation is in the global reference frame.
std::vector<Vector3d>
      RotateRelativeTranslationsToGlobalFrame(
          const std::unordered_map<image_t, Vector4d>& orientations,
          const std::vector<ImagePair>& view_pairs) {
  std::vector<Vector3d> rotated_translations;
  rotated_translations.reserve(orientations.size());

  for (const auto& view_pair : view_pairs) {
    const Vector4d view_to_world_rotation =
        InvertQuaternion(theia::FindOrDie(orientations, view_pair.image_id2));
    Vector3d rotated_translation = QuaternionRotatePoint(view_to_world_rotation,
                                                         view_pair.tvec);
    rotated_translations.push_back(rotated_translation);
  }
  return rotated_translations;
}

// Find the next view to add to the order. We attempt to choose a source (i.e.,
// a node with no incoming edges) or choose a node based on a heuristic such
// that it has the most source-like properties.
image_t FindNextViewInOrder(
    const std::unordered_map<image_t, MFASNode>& degrees_for_view) {
  image_t best_choice = kInvalidImageId;
  double best_score = 0;
  for (const auto& view : degrees_for_view) {
    // If the view is a source view, return it.
    if (view.second.incoming_nodes.size() == 0) {
      return view.first;
    }

    // Otherwise, keep track of the max score seen so far.
    const double score = (view.second.outgoing_weight + 1.0) /
                         (view.second.incoming_weight + 1.0);
    if (score > best_score) {
      best_choice = view.first;
      best_score = score;
    }
  }

  return best_choice;
}

// Based on the 1D translation projections, compute an ordering of the
// translations.
std::unordered_map<image_t, int> OrderTranslationsFromProjections(
    const std::vector<ImagePair>& view_pairs,
    const std::vector<double>& translation_direction_projections) {
  // Compute the degrees of all vertices as the sum of weights coming in or out.
  std::unordered_map<image_t, MFASNode> degrees_for_view;
  for (size_t i = 0; i < view_pairs.size(); ++i) {
    ImagePair view_pair(view_pairs[i]);
    if (translation_direction_projections[i] <= 0) {
      std::swap(view_pair.image_id1, view_pair.image_id2);
    }

    // Update the MFAS entry.
    const double weight = std::abs(translation_direction_projections[i]);
    degrees_for_view[view_pair.image_id2].incoming_weight += weight;
    degrees_for_view[view_pair.image_id1].outgoing_weight += weight;
    degrees_for_view[view_pair.image_id2].incoming_nodes.emplace(
        view_pair.image_id1, weight);
    degrees_for_view[view_pair.image_id1].outgoing_nodes.emplace(
        view_pair.image_id2, weight);
  }

  // Compute the ordering.
  const int num_views = degrees_for_view.size();
  std::unordered_map<image_t, int> translation_ordering;
  for (int i = 0; i < num_views; i++) {
    // Find the next view to add.
    const image_t next_view_in_order = FindNextViewInOrder(degrees_for_view);
    translation_ordering[next_view_in_order] = i;

    // Update the MFAS graph and remove the next view from the degrees_for_view.
    const auto& next_view_info =
        theia::FindOrDie(degrees_for_view, next_view_in_order);
    for (auto& neighbor_info : next_view_info.incoming_nodes) {
      degrees_for_view[neighbor_info.first].outgoing_weight -=
          neighbor_info.second;
      degrees_for_view[neighbor_info.first].outgoing_nodes.erase(
          next_view_in_order);
    }
    for (auto& neighbor_info : next_view_info.outgoing_nodes) {
      degrees_for_view[neighbor_info.first].incoming_weight -=
          neighbor_info.second;
      degrees_for_view[neighbor_info.first].incoming_nodes.erase(
          next_view_in_order);
    }
    degrees_for_view.erase(next_view_in_order);
  }

  return translation_ordering;
}

// Projects all the of the translation onto the given axis.
std::vector<double> ProjectTranslationsOntoAxis(
    const Vector3d& axis,
    const std::vector<Vector3d>& relative_translations) {
  std::vector<double> projection_weights;
  projection_weights.reserve(relative_translations.size());

  for (const auto& relative_translation : relative_translations) {
    const double projection_weight = relative_translation.dot(axis);
    projection_weights.push_back(projection_weight);
  }
  return projection_weights;
}

// This chooses a random axis based on the given relative translations.
void ComputeMeanVariance(
    const std::vector<Vector3d>& relative_translations,
    Vector3d* mean,
    Vector3d* variance) {
  mean->setZero();
  variance->setZero();
  for (const auto& translation : relative_translations) {
    *mean += translation;
  }
  *mean /= static_cast<double>(relative_translations.size());

  for (const auto& translation : relative_translations) {
    *variance += (translation - *mean).cwiseAbs2();
  }
  *variance /= static_cast<double>(relative_translations.size() - 1);
}

// Performs a single iterations of the translation filtering. This method is
// thread-safe.
void TranslationFilteringIteration(
    const std::vector<ImagePair>& view_pairs,
    const std::vector<Vector3d>& relative_translations,
    const Vector3d& direction_mean,
    const Vector3d& direction_variance,
    const std::shared_ptr<theia::RandomNumberGenerator>& rng,
    std::mutex* mutex,
    std::vector<double>* bad_edge_weight) {
  // Create the random number generator within each thread. If the random number
  // generator is not supplied then create a new one within each thread.
  std::shared_ptr<theia::RandomNumberGenerator> local_rng;
  if (rng.get() == nullptr) {
    local_rng = std::make_shared<theia::RandomNumberGenerator>();
  } else {
    local_rng = rng;
  }

  // Get a random vector to project all relative translations on to.
  const Vector3d random_axis =
      Vector3d(
          local_rng->RandGaussian(direction_mean[0], direction_variance[0]),
          local_rng->RandGaussian(direction_mean[1], direction_variance[1]),
          local_rng->RandGaussian(direction_mean[2], direction_variance[2]))
          .normalized();

  // Project all vectors.
  std::vector<double> translation_direction_projections =
          ProjectTranslationsOntoAxis(random_axis, relative_translations);

  // Compute ordering.
  const std::unordered_map<image_t, int>& translation_ordering =
      OrderTranslationsFromProjections(view_pairs, translation_direction_projections);

  // Compute bad edge weights.
  for (size_t i = 0; i < view_pairs.size(); ++i) {
    const int ordering_diff =
        theia::FindOrDie(translation_ordering, view_pairs[i].image_id2) -
        theia::FindOrDie(translation_ordering, view_pairs[i].image_id1);
    const double projection_weight_of_edge = translation_direction_projections[i];

    DLOG(INFO) << "Edge (" << view_pairs[i].image_id1 << ", " << view_pairs[i].image_id2
            << ") has ordering diff of " << ordering_diff
            << " and a projection of " << projection_weight_of_edge << " from "
            << relative_translations[i].transpose();
    // If the ordering is inconsistent, add the absolute value of the bad weight
    // to the aggregate bad weight.
    if ((ordering_diff < 0 && projection_weight_of_edge > 0) ||
        (ordering_diff > 0 && projection_weight_of_edge < 0)) {
      std::lock_guard<std::mutex> lock(*mutex);
      (*bad_edge_weight)[i] += std::abs(projection_weight_of_edge);
    }
  }
}

}  // namespace

void FilterViewPairsFromOrientation(
    const std::unordered_map<image_t, Eigen::Vector4d>& orientations,
    const double max_relative_rotation_difference_degrees,
    std::vector<ImagePair>* view_pairs) {
  CHECK_GE(max_relative_rotation_difference_degrees, 0.0);

  // Precompute the squared threshold in radians.
  const double max_relative_rotation_difference_radians =
      DegToRad(max_relative_rotation_difference_degrees);
  const double sq_max_relative_rotation_difference_radians =
      max_relative_rotation_difference_radians *
      max_relative_rotation_difference_radians;

  std::vector<ImagePair> filtered_view_pairs;
  std::unordered_set<size_t> removed_ids;
  int remove_count = 0;
  for (size_t i = 0; i < view_pairs->size(); ++i) {
    const auto& view_pair = (*view_pairs)[i];
    const Eigen::Vector4d* orientation1 =
        theia::FindOrNull(orientations, view_pair.image_id1);
    const Eigen::Vector4d* orientation2 =
        theia::FindOrNull(orientations, view_pair.image_id2);
    bool remove = false;

    // If the view pair contains a view that does not have an orientation then
    // remove it.
    if (orientation1 == nullptr || orientation2 == nullptr) {
      LOG(WARNING)
          << "View pair (" << view_pair.image_id1 << ", "
          << view_pair.image_id2
          << ") contains a view that does not exist! Removing the view pair.";
      remove = true;
    } else {
      // Remove the view pair if the relative rotation estimate is not within the
      // tolerance.
      if (!AngularDifferenceIsAcceptable(
              *orientation1,
              *orientation2,
              view_pair.qvec,
              sq_max_relative_rotation_difference_radians)) {
        remove = true;
      }
    }

    if (remove) {
      ++remove_count;
    } else {
      filtered_view_pairs.push_back(view_pair);
    }
  }

  LOG(INFO) << "Removed " << remove_count
          << " view pairs by rotation filtering.";
  std::swap(*view_pairs, filtered_view_pairs);
}

void FilterViewPairsFromRelativeTranslation(
    const FilterViewPairsFromRelativeTranslationOptions& options,
    const std::unordered_map<image_t, Vector4d>& orientations,
    std::vector<ImagePair>* view_pairs) {
  // Weights of edges that have been accumulated throughout the iterations. A
  // higher weight means the edge is more likely to be bad.
  std::vector<double> bad_edge_weight(view_pairs->size(), 0.0);

  // Compute the adjusted translations so that they are oriented in the global
  // frame.
  std::vector<Vector3d> rotated_translations =
      RotateRelativeTranslationsToGlobalFrame(orientations, *view_pairs);

  Vector3d translation_mean, translation_variance;
  ComputeMeanVariance(rotated_translations,
                      &translation_mean,
                      &translation_variance);

  std::unique_ptr<ThreadPool> pool(new ThreadPool(options.num_threads));
  std::mutex mutex;
  for (int i = 0; i < options.num_iterations; i++) {
    pool->AddTask(TranslationFilteringIteration,
              *view_pairs,
              rotated_translations,
              translation_mean,
              translation_variance,
              options.rng,
              &mutex,
              &bad_edge_weight);
  }
  // Wait for tasks to finish.
  pool->Wait();
  pool.reset(nullptr);

  // Remove all the bad edges.
  std::vector<ImagePair> filtered_view_pairs;
  const double max_aggregated_projection_tolerance =
      options.translation_projection_tolerance * options.num_iterations;
  int num_view_pairs_removed = 0;
  for (size_t i = 0; i < view_pairs->size(); ++i) {
    const auto& view_pair = (*view_pairs)[i];
    DLOG(INFO) << "View pair (" << view_pair.image_id1 << ", "
            << view_pair.image_id2 << ") projection = " << bad_edge_weight[i];
    if (bad_edge_weight[i] > max_aggregated_projection_tolerance) {
      ++num_view_pairs_removed;
    } else {
      filtered_view_pairs.push_back(view_pair);
    }
  }

  LOG(INFO) << "Removed " << num_view_pairs_removed
          << " view pairs by relative translation filtering.";
  std::swap(*view_pairs, filtered_view_pairs);
}

void RemoveDisconnectedViewPairs(std::vector<ImagePair>& pairs, bool only_adjacent) {
  // Extract all connected components.
    theia::ConnectedComponents<image_t> cc_extractor;
  for (const auto& pair : pairs) {
    if (!only_adjacent || pair.image_id1 == pair.image_id2 + 1 ||
        pair.image_id2 == pair.image_id1 + 1) {
      cc_extractor.AddEdge(pair.image_id1, pair.image_id2);
    }
  }
  std::unordered_map<image_t, std::unordered_set<image_t>> connected_components;
  cc_extractor.Extract(&connected_components);

  // Find the largest connected component.
  int max_cc_size = 0;
  image_t largest_cc_root_id = kInvalidImageId;
  int total_node = 0;
  for (const auto& connected_component : connected_components) {
    total_node += connected_component.second.size();
    if (connected_component.second.size() > max_cc_size) {
      max_cc_size = connected_component.second.size();
      largest_cc_root_id = connected_component.first;
    }
  }
  LOG(INFO) << "Filtered image: " << total_node << " -> " << max_cc_size;

  // Remove all view pairs containing a view to remove (i.e. the ones that are
  // not in the largest connectedcomponent).
  std::unordered_set<image_t> removed_views;
  for (const auto& connected_component : connected_components) {
    if (connected_component.first == largest_cc_root_id) {
      continue;
    }

    // NOTE: The connected component will contain the root id as well, so we do
    // not explicity have to remove connected_component.first since it will
    // exist in connected_components.second
    for (const image_t view_id2 : connected_component.second) {
      removed_views.insert(view_id2);
    }
  }

  std::vector<ImagePair> filtered_pairs;
  for (const auto& pair : pairs) {
    if (removed_views.count(pair.image_id1) == 0 && removed_views.count(pair.image_id2) == 0) {
      filtered_pairs.push_back(pair);
    }
  }
  LOG(INFO) << "Filtered edge: " << pairs.size() << " -> " << filtered_pairs.size();
  std::swap(pairs, filtered_pairs);
}

}
