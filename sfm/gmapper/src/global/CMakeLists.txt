set(FOLDER_NAME "global")

GCOLMAP_ADD_SOURCES(
  rotation_estimator.h
  robust_rotation_estimator.h robust_rotation_estimator.cc
  position_estimator.h
  linear_position_estimator.h linear_position_estimator.cc
  nonlinear_position_estimator.h nonlinear_position_estimator.cc
  orientation_util.h orientation_util.cc
  filter_util.h filter_util.cc
  known_rotation_util.h known_rotation_util.cc
  triplet_util.h triplet_util.cc
  image_triplet.h
  least_unsquared_deviation_position_estimator.h least_unsquared_deviation_position_estimator.cc
)

