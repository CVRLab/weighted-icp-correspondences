#pragma once

#include <limits>
#include <utility>
#include <vector>

#include <pcl/correspondence.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/types.h>

template <typename PointSource, typename PointTarget, typename Scalar = float>
class WeightedCorrespondenceEstimation
    : public pcl::registration::CorrespondenceEstimation<PointSource, PointTarget, Scalar> {
 public:
  using ParentCorrespondenceEstimation =
      pcl::registration::CorrespondenceEstimation<PointSource, PointTarget, Scalar>;

  using ParentCorrespondenceEstimation::indices_;
  using ParentCorrespondenceEstimation::input_;
  using ParentCorrespondenceEstimation::tree_;

 public:
  WeightedCorrespondenceEstimation(size_t k = 1) : k(k) {}

  void setSourceWeights(const std::vector<double>& weights) { source_weights = weights; }
  void setTargetWeights(const std::vector<double>& weights) { target_weights = weights; }

  void determineCorrespondences(
      pcl::Correspondences& correspondences,
      double max_distance = std::numeric_limits<double>::max()
  ) override {
    if (!ParentCorrespondenceEstimation::initCompute())
      return;

    correspondences.resize(indices_->size());

    pcl::Indices candidate_indices(k);
    std::vector<float> candidate_distances(k);
    pcl::Correspondence corr;
    unsigned int nr_valid_correspondences = 0;
    double max_dist_sqr = max_distance * max_distance;

    // Iterate over the input set of source indices
    for (const auto& idx : (*indices_)) {
      // Check if the template types are the same. If true, avoid a copy.
      const auto& pt =
          pcl::registration::detail::pointCopyOrRef<PointTarget, PointSource>(input_, idx);
      tree_->nearestKSearch(pt, k, candidate_indices, candidate_distances);

      // Select correspondence with the heaviest weight
      size_t best_candidate = 0;
      double best_weight = 0;
      for (size_t i = 0; i < k; ++i) {
        if (candidate_distances[i] > max_dist_sqr)
          break;

        auto weight = target_weights[candidate_indices[i]];
        if (weight > best_weight) {
          best_candidate = i;
          best_weight = weight;
        }
      }

      if (candidate_distances[best_candidate] > max_dist_sqr || best_weight == 0)
        continue;

      corr.index_query = idx;
      corr.index_match = candidate_indices[best_candidate];
      corr.distance = candidate_distances[best_candidate];
      correspondences[nr_valid_correspondences++] = corr;
    }

    correspondences.resize(nr_valid_correspondences);
    ParentCorrespondenceEstimation::deinitCompute();
  }

  void determineReciprocalCorrespondences(
      pcl::Correspondences& correspondences,
      double max_distance
  ) override {
    PCL_WARN("Reciprocal correspondences are not implemented.");
    return;
  }

 private:
  size_t k;
  std::vector<double> source_weights;
  std::vector<double> target_weights;
};
