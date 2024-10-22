#include <array>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <mrpt/containers/yaml.h>
#include <mrpt/img/TPixelCoord.h>
#include <mrpt/io/CTextFileLinesParser.h>
#include <mrpt/math/CQuaternion.h>
#include <mrpt/poses/CPoint3D.h>
#include <mrpt/poses/CPointPDFGaussian.h>
#include <mrpt/poses/CPose3D.h>
#include <mrpt/poses/CPose3DQuat.h>
#include <mrpt/system/filesystem.h>
#include <pcl/console/print.h>
#include <pcl/console/time.h>
#include <pcl/correspondence.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation.h>
#include <pcl/types.h>
#include <pcl/visualization/pcl_plotter.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <opencv2/core/utility.hpp>

// Global variables
mrpt::containers::yaml config;
pcl::console::TicToc tic_toc;

// Global definitions
typedef pcl::PointXYZ Point;
typedef pcl::PointCloud<Point> PointCloud;

struct Landmark {
 public:
  typedef std::shared_ptr<Landmark> Ptr;
  Landmark() = default;

  size_t id;
  mrpt::poses::CPointPDFGaussian position;
  size_t num_observations = 0;
  std::vector<std::pair<size_t, mrpt::img::TPixelCoordf>> observations;

  double mean_observation_distance = 0;
  double maximum_observation_angle = 0;
};

struct Map {
  typedef std::shared_ptr<Map> Ptr;

  std::map<size_t, size_t> frame_id_map;
  std::vector<mrpt::poses::CPose3D> frames;

  std::map<size_t, size_t> landmark_id_map;
  std::vector<Landmark::Ptr> landmarks;

  PointCloud::Ptr cloud;

  void load(const std::string& path) {
    if (!mrpt::system::fileExists(path))
      throw std::runtime_error("Map file does not exist: " + path);

    mrpt::io::CTextFileLinesParser parser(path);
    std::istringstream line;
    while (parser.getNextLine(line)) {
      std::string type;
      line >> type;

      if (type == "FRAME") {
        size_t id;
        double t[3], q[4];
        line >> id >> t[0] >> t[1] >> t[2] >> q[0] >> q[1] >> q[2] >> q[3];

        mrpt::poses::CPose3DQuat pose(
            t[0], t[1], t[2], mrpt::math::CQuaternionDouble(q[0], q[1], q[2], q[3])
        );

        frame_id_map[id] = frames.size();
        frames.push_back(mrpt::poses::CPose3D(pose));

      } else if (type == "LANDMARK") {
        size_t id, num_observations;
        double p[3], cov[3][3];

        line >> id;
        line >> p[0] >> p[1] >> p[2];
        line >> cov[0][0] >> cov[0][1] >> cov[0][2] >> cov[1][0] >> cov[1][1] >> cov[1][2] >>
            cov[2][0] >> cov[2][1] >> cov[2][2];
        line >> num_observations;

        if (num_observations < 2)
          continue;

        Landmark::Ptr landmark(new Landmark);
        landmark->id = id;
        landmark->position.mean = mrpt::poses::CPoint3D(p[0], p[1], p[2]);
        landmark->position.cov(0, 0) = cov[0][0];
        landmark->position.cov(0, 1) = cov[0][1];
        landmark->position.cov(0, 2) = cov[0][2];
        landmark->position.cov(1, 0) = cov[1][0];
        landmark->position.cov(1, 1) = cov[1][1];
        landmark->position.cov(1, 2) = cov[1][2];
        landmark->position.cov(2, 0) = cov[2][0];
        landmark->position.cov(2, 1) = cov[2][1];
        landmark->position.cov(2, 2) = cov[2][2];
        landmark->num_observations = num_observations;

        for (size_t i = 0; i < num_observations; i++) {
          size_t frame_id;
          double x, y;
          line >> frame_id >> x >> y;
          landmark->observations.push_back({frame_id, mrpt::img::TPixelCoordf(x, y)});
        }

        landmark_id_map[id] = landmarks.size();
        landmarks.push_back(landmark);
      } else {
        throw std::runtime_error("Unknown entity type: " + type);
      }
    }

    cloud.reset(new PointCloud(landmarks.size(), 1));
    for (size_t i = 0; i < landmarks.size(); i++) {
      auto landmark = landmarks[i];
      (*cloud)[i].x = landmark->position.mean.x();
      (*cloud)[i].y = landmark->position.mean.y();
      (*cloud)[i].z = landmark->position.mean.z();

      double sum = 0;
      for (auto& [frame_id, pixel_coord] : landmark->observations) {
        auto frame_index = frame_id_map[frame_id];
        auto frame = frames[frame_index];
        auto direction = frame.translation() - landmark->position.mean.asTPoint();
        sum += direction.norm();
      }
      landmark->mean_observation_distance = sum / landmark->num_observations;
    }

    // std::cout << "Loaded " << frames.size() << " frames and " << landmarks.size() << "landmarks."
    //           << std::endl;
  }
};

struct Experiment {
  typedef std::shared_ptr<Experiment> Ptr;

  std::string name;

  Map::Ptr source_map;
  Map::Ptr target_map;

  PointCloud::Ptr cloud_result;
  pcl::IterativeClosestPoint<Point, Point, double>::Ptr icp;
  std::vector<double> score_history;
  double duration = 0;

  int viewport_id;
  std::array<uint8_t, 3> color = {255, 255, 255};
};

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

template <typename PointSource, typename PointTarget, typename Scalar = float>
class IdentityTranformEstimation
    : public pcl::registration::TransformationEstimation<PointSource, PointTarget, Scalar> {
 public:
  using Matrix4 =
      pcl::registration::TransformationEstimation<PointSource, PointTarget, Scalar>::Matrix4;

  void estimateRigidTransformation(
      const pcl::PointCloud<PointSource>& cloud_src,
      const pcl::PointCloud<PointTarget>& cloud_tgt,
      Matrix4& transformation_matrix
  ) const {
    transformation_matrix = Matrix4::Identity();
  }

  void estimateRigidTransformation(
      const pcl::PointCloud<PointSource>& cloud_src,
      const pcl::Indices& indices_src,
      const pcl::PointCloud<PointTarget>& cloud_tgt,
      Matrix4& transformation_matrix
  ) const {
    transformation_matrix = Matrix4::Identity();
  }

  void estimateRigidTransformation(
      const pcl::PointCloud<PointSource>& cloud_src,
      const pcl::Indices& indices_src,
      const pcl::PointCloud<PointTarget>& cloud_tgt,
      const pcl::Indices& indices_tgt,
      Matrix4& transformation_matrix
  ) const {
    transformation_matrix = Matrix4::Identity();
  }

  void estimateRigidTransformation(
      const pcl::PointCloud<PointSource>& cloud_src,
      const pcl::PointCloud<PointTarget>& cloud_tgt,
      const pcl::Correspondences& correspondences,
      Matrix4& transformation_matrix
  ) const {
    transformation_matrix = Matrix4::Identity();
  }
};

std::vector<Experiment::Ptr> run_experiments(Map::Ptr source_map, Map::Ptr target_map) {
  using ICP = pcl::IterativeClosestPoint<Point, Point, double>;

  auto experiment_toggle = config["experiment_toggle"];
  auto run_initial = experiment_toggle.getOrDefault<bool>("initial", true);
  auto run_base = experiment_toggle.getOrDefault<bool>("base", true);
  auto run_covariance = experiment_toggle.getOrDefault<bool>("covariance", true);
  auto run_num_observations = experiment_toggle.getOrDefault<bool>("num_observations", true);
  auto run_distance_mean = experiment_toggle.getOrDefault<bool>("distance_mean", true);

  auto max_iterations = config.getOrDefault<size_t>("max_iterations", 100);
  auto correspondence_search_knn = config.getOrDefault<size_t>("correspondence_search_knn", 1);

  std::vector<Experiment::Ptr> experiments;

  // Initial experiment
  if (run_initial) {
    Experiment::Ptr experiment(new Experiment);
    experiment->name = "Initial";
    experiment->color = {255, 0, 0};
    experiment->source_map = source_map;
    experiment->target_map = target_map;

    // Setup ICP
    experiment->cloud_result.reset(new PointCloud);
    experiment->icp.reset(new ICP);
    experiment->icp->setMaximumIterations(1);
    experiment->icp->setInputTarget(target_map->cloud);
    experiment->icp->setInputSource(source_map->cloud);

    // Configure ICP steps
    experiment->icp->setTransformationEstimation(
        std::make_shared<IdentityTranformEstimation<Point, Point, double>>()
    );

    // Register visualization callback
    std::function<ICP::UpdateVisualizerCallbackSignature> visualizer_callback =
        [&](const PointCloud&, const pcl::Indices&, const PointCloud&, const pcl::Indices&) {
          auto score = experiment->icp->getFitnessScore();
          if (score == 0 && experiment->score_history.empty())
            return;
          experiment->score_history.push_back(score);
          std::cout << experiment->name << " iteration " << experiment->icp->nr_iterations_
                    << " score: " << score << std::endl;
        };
    experiment->icp->registerVisualizationCallback(visualizer_callback);

    // Run ICP alignment
    std::cout << std::endl << "Running experiment " << experiment->name << "..." << std::endl;
    tic_toc.tic();
    experiment->icp->align(*(experiment->cloud_result));
    experiment->duration = tic_toc.toc() / 1000.0;
    std::cout << "Experiment " << experiment->name << " finished after " << experiment->duration
              << "s with criteria " << experiment->icp->convergence_criteria_->getConvergenceState()
              << std::endl;

    experiments.push_back(experiment);
  }

  // Base experiment
  if (run_base) {
    Experiment::Ptr experiment(new Experiment);
    experiment->name = "Base";
    experiment->color = {0, 255, 0};
    experiment->source_map = source_map;
    experiment->target_map = target_map;

    // Setup ICP
    experiment->cloud_result.reset(new PointCloud);
    experiment->icp.reset(new ICP);
    experiment->icp->setMaximumIterations(max_iterations);
    experiment->icp->setInputTarget(target_map->cloud);
    experiment->icp->setInputSource(source_map->cloud);

    // Register visualization callback
    std::function<ICP::UpdateVisualizerCallbackSignature> visualizer_callback =
        [&](const PointCloud&, const pcl::Indices&, const PointCloud&, const pcl::Indices&) {
          auto score = experiment->icp->getFitnessScore();
          if (score == 0 && experiment->score_history.empty())
            return;
          experiment->score_history.push_back(score);
          std::cout << experiment->name << " iteration " << experiment->icp->nr_iterations_
                    << " score: " << score << std::endl;
        };
    experiment->icp->registerVisualizationCallback(visualizer_callback);

    // Run ICP alignment
    std::cout << std::endl << "Running experiment " << experiment->name << "..." << std::endl;
    tic_toc.tic();
    experiment->icp->align(*(experiment->cloud_result));
    experiment->duration = tic_toc.toc() / 1000.0;
    std::cout << "Experiment " << experiment->name << " finished after " << experiment->duration
              << "s with criteria " << experiment->icp->convergence_criteria_->getConvergenceState()
              << std::endl;

    experiments.push_back(experiment);
  }

  // Covariance experiment
  if (run_covariance) {
    Experiment::Ptr experiment(new Experiment);
    experiment->name = "Covariance";
    experiment->color = {0, 0, 255};
    experiment->source_map = source_map;
    experiment->target_map = target_map;

    // Setup ICP
    experiment->cloud_result.reset(new PointCloud);
    experiment->icp.reset(new ICP);
    experiment->icp->setMaximumIterations(max_iterations);
    experiment->icp->setInputTarget(target_map->cloud);
    experiment->icp->setInputSource(source_map->cloud);

    // Compute point weights
    std::vector<double> source_weights(source_map->cloud->size());
    for (size_t i = 0; i < source_map->cloud->size(); i++)
      source_weights[i] = 1.0 / (1.0 + source_map->landmarks[i]->position.cov.norm());

    std::vector<double> target_weights(target_map->cloud->size());
    for (size_t i = 0; i < target_map->cloud->size(); i++)
      target_weights[i] = 1.0 / (1.0 + target_map->landmarks[i]->position.cov.norm());

    // Configure ICP steps
    auto correspondence_estimation =
        std::make_shared<WeightedCorrespondenceEstimation<Point, Point, double>>(
            correspondence_search_knn
        );
    correspondence_estimation->setSourceWeights(source_weights);
    correspondence_estimation->setTargetWeights(target_weights);
    experiment->icp->setCorrespondenceEstimation(correspondence_estimation);

    // Register visualization callback
    std::function<ICP::UpdateVisualizerCallbackSignature> visualizer_callback =
        [&](const PointCloud&, const pcl::Indices&, const PointCloud&, const pcl::Indices&) {
          auto score = experiment->icp->getFitnessScore();
          if (score == 0 && experiment->score_history.empty())
            return;
          experiment->score_history.push_back(score);
          std::cout << experiment->name << " iteration " << experiment->icp->nr_iterations_
                    << " score: " << score << std::endl;
        };
    experiment->icp->registerVisualizationCallback(visualizer_callback);

    // Run ICP alignment
    std::cout << std::endl << "Running experiment " << experiment->name << "..." << std::endl;
    tic_toc.tic();
    experiment->icp->align(*(experiment->cloud_result));
    experiment->duration = tic_toc.toc() / 1000.0;
    std::cout << "Experiment " << experiment->name << " finished after " << experiment->duration
              << "s with criteria " << experiment->icp->convergence_criteria_->getConvergenceState()
              << std::endl;

    experiments.push_back(experiment);
  }

  // Num observations experiment
  if (run_num_observations) {
    Experiment::Ptr experiment(new Experiment);
    experiment->name = "Num observations";
    experiment->color = {255, 0, 255};
    experiment->source_map = source_map;
    experiment->target_map = target_map;

    // Setup ICP
    experiment->cloud_result.reset(new PointCloud);
    experiment->icp.reset(new ICP);
    experiment->icp->setMaximumIterations(max_iterations);
    experiment->icp->setInputTarget(target_map->cloud);
    experiment->icp->setInputSource(source_map->cloud);

    // Compute point weights
    std::vector<double> source_weights(source_map->cloud->size());
    for (size_t i = 0; i < source_map->cloud->size(); i++)
      source_weights[i] = source_map->landmarks[i]->num_observations;

    std::vector<double> target_weights(target_map->cloud->size());
    for (size_t i = 0; i < target_map->cloud->size(); i++)
      target_weights[i] = target_map->landmarks[i]->num_observations;

    // Configure ICP steps
    auto correspondence_estimation =
        std::make_shared<WeightedCorrespondenceEstimation<Point, Point, double>>(
            correspondence_search_knn
        );
    correspondence_estimation->setSourceWeights(source_weights);
    correspondence_estimation->setTargetWeights(target_weights);
    experiment->icp->setCorrespondenceEstimation(correspondence_estimation);

    // Register visualization callback
    std::function<ICP::UpdateVisualizerCallbackSignature> visualizer_callback =
        [&](const PointCloud&, const pcl::Indices&, const PointCloud&, const pcl::Indices&) {
          auto score = experiment->icp->getFitnessScore();
          if (score == 0 && experiment->score_history.empty())
            return;
          experiment->score_history.push_back(score);
          std::cout << experiment->name << " iteration " << experiment->icp->nr_iterations_
                    << " score: " << score << std::endl;
        };
    experiment->icp->registerVisualizationCallback(visualizer_callback);

    // Run ICP alignment
    std::cout << std::endl << "Running experiment " << experiment->name << "..." << std::endl;
    tic_toc.tic();
    experiment->icp->align(*(experiment->cloud_result));
    experiment->duration = tic_toc.toc() / 1000.0;
    std::cout << "Experiment " << experiment->name << " finished after " << experiment->duration
              << "s with criteria " << experiment->icp->convergence_criteria_->getConvergenceState()
              << std::endl;

    experiments.push_back(experiment);
  }

  // Distance mean experiment
  if (run_distance_mean) {
    Experiment::Ptr experiment(new Experiment);
    experiment->name = "Distance mean";
    experiment->color = {0, 255, 255};
    experiment->source_map = source_map;
    experiment->target_map = target_map;

    // Setup ICP
    experiment->cloud_result.reset(new PointCloud);
    experiment->icp.reset(new ICP);
    experiment->icp->setMaximumIterations(max_iterations);
    experiment->icp->setInputTarget(target_map->cloud);
    experiment->icp->setInputSource(source_map->cloud);

    // Compute point weights
    std::vector<double> source_weights(source_map->cloud->size());
    for (size_t i = 0; i < source_map->cloud->size(); i++)
      source_weights[i] = 1.0 / (1.0 + source_map->landmarks[i]->mean_observation_distance);

    std::vector<double> target_weights(target_map->cloud->size());
    for (size_t i = 0; i < target_map->cloud->size(); i++)
      target_weights[i] = 1.0 / (1.0 + target_map->landmarks[i]->mean_observation_distance);

    // Configure ICP steps
    auto correspondence_estimation =
        std::make_shared<WeightedCorrespondenceEstimation<Point, Point, double>>(
            correspondence_search_knn
        );
    correspondence_estimation->setSourceWeights(source_weights);
    correspondence_estimation->setTargetWeights(target_weights);
    experiment->icp->setCorrespondenceEstimation(correspondence_estimation);

    // Register visualization callback
    std::function<ICP::UpdateVisualizerCallbackSignature> visualizer_callback =
        [&](const PointCloud&, const pcl::Indices&, const PointCloud&, const pcl::Indices&) {
          auto score = experiment->icp->getFitnessScore();
          if (score == 0 && experiment->score_history.empty())
            return;
          experiment->score_history.push_back(score);
          std::cout << experiment->name << " iteration " << experiment->icp->nr_iterations_
                    << " score: " << score << std::endl;
        };
    experiment->icp->registerVisualizationCallback(visualizer_callback);

    // Run ICP alignment
    std::cout << std::endl << "Running experiment " << experiment->name << "..." << std::endl;
    tic_toc.tic();
    experiment->icp->align(*(experiment->cloud_result));
    experiment->duration = tic_toc.toc() / 1000.0;
    std::cout << "Experiment " << experiment->name << " finished after " << experiment->duration
              << "s with criteria " << experiment->icp->convergence_criteria_->getConvergenceState()
              << std::endl;

    experiments.push_back(experiment);
  }

  return experiments;
}

void alignment_visualization(std::vector<Experiment::Ptr> experiments) {
  using ColorHandler = pcl::visualization::PointCloudColorHandlerCustom<Point>;

  // float bg_gray = 0.2;
  float bg_gray = 0.8;
  float text_gray = 1 - bg_gray;

  bool show_info = true;

  pcl::visualization::PCLVisualizer viewer("HVSLAM ICP Map Alignment");

  size_t num_experiments = experiments.size();
  size_t top_experiments = std::ceil(num_experiments / 2.0);
  size_t bottom_experiments = num_experiments - top_experiments;
  float y_step = bottom_experiments > 0 ? 0.5 : 0.0;

  for (size_t i = 0; i < top_experiments; i++) {
    auto experiment = experiments[i];

    float xmin = i / (float)top_experiments;
    float xmax = (i + 1) / (float)top_experiments;

    int viewport_id = i + 1;
    experiment->viewport_id = viewport_id;
    viewer.createViewPort(xmin, y_step, xmax, 1.0, viewport_id);
  }

  for (size_t i = 0; i < bottom_experiments; i++) {
    auto experiment = experiments[top_experiments + i];

    float xmin = i / (float)bottom_experiments;
    float xmax = (i + 1) / (float)bottom_experiments;

    int viewport_id = top_experiments + i + 1;
    experiment->viewport_id = viewport_id;
    viewer.createViewPort(xmin, 0.0, xmax, y_step, viewport_id);
  }

  for (size_t i = 0; i < num_experiments; i++) {
    auto experiment = experiments[i];

    std::ostringstream vid, converged, score, iterations, duration;
    vid << "viewport_" << experiment->viewport_id;
    converged << "Converged: " << experiment->icp->hasConverged();
    score << "Score: " << experiment->icp->getFitnessScore();
    iterations << "Iterations: " << experiment->icp->nr_iterations_;
    duration << "Duration: " << experiment->duration << " s";

    std::string vid_prefix = vid.str();

    if (show_info) {
      viewer.addText(
          experiment->name,
          10,
          70,
          16,
          text_gray,
          text_gray,
          text_gray,
          vid_prefix + "_name",
          experiment->viewport_id
      );
      viewer.addText(
          converged.str(),
          10,
          50,
          16,
          text_gray,
          text_gray,
          text_gray,
          vid_prefix + "_converged",
          experiment->viewport_id
      );
      viewer.addText(
          score.str(),
          10,
          35,
          16,
          text_gray,
          text_gray,
          text_gray,
          vid_prefix + "_score",
          experiment->viewport_id
      );
      viewer.addText(
          iterations.str(),
          10,
          20,
          16,
          text_gray,
          text_gray,
          text_gray,
          vid_prefix + "_iterations",
          experiment->viewport_id
      );
      viewer.addText(
          duration.str(),
          10,
          5,
          16,
          text_gray,
          text_gray,
          text_gray,
          vid_prefix + "_duration",
          experiment->viewport_id
      );
    }

    viewer.addPointCloud(
        experiment->target_map->cloud,
        ColorHandler(experiment->target_map->cloud, text_gray, text_gray, text_gray),
        vid_prefix + "_target",
        experiment->viewport_id
    );
    viewer.setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
        2,
        vid_prefix + "_target",
        experiment->viewport_id
    );

    viewer.addPointCloud(
        experiment->cloud_result,
        ColorHandler(
            experiment->cloud_result,
            experiment->color[0],
            experiment->color[1],
            experiment->color[2]
        ),
        vid_prefix + "_result",
        experiment->viewport_id
    );
    viewer.setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
        2,
        vid_prefix + "_result",
        experiment->viewport_id
    );
  }

  // Final viewer setup
  viewer.setBackgroundColor(bg_gray, bg_gray, bg_gray);
  viewer.setSize(1280, 960);
  // viewer.addCoordinateSystem();

  viewer.spin();
  viewer.close();
}

void score_histories_visualization(std::vector<Experiment::Ptr> experiments) {
  using PlotPoint = std::pair<double, double>;

  pcl::visualization::PCLPlotter plotter;
  plotter.setShowLegend(true);
  plotter.setXTitle("Iteration");
  plotter.setYTitle("Score");

  size_t max_history_size = 0;
  for (auto& experiment : experiments) {
    auto history_size = experiment->score_history.size();
    if (history_size > max_history_size)
      max_history_size = history_size;
  }

  for (auto& experiment : experiments) {
    if (experiment->score_history.size() == 1) {
      continue;
      for (size_t i = experiment->score_history.size(); i < max_history_size; i++)
        experiment->score_history.push_back(experiment->score_history[i - 1]);
    }

    std::vector<PlotPoint> data;
    for (size_t i = 0; i < experiment->score_history.size(); i++)
      data.push_back({i + 1, experiment->score_history[i]});

    plotter.addPlotData(data, experiment->name.c_str());
  }

  plotter.setXRange(0, max_history_size + 1);
  plotter.plot();
  plotter.close();
}

int main(int argc, char** argv) {
  try {
    pcl::console::print_highlight("Map Alignment v1.0.0\n");
    pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);

    // Parse command line arguments
    const std::string keys =
        "{ @source_map | | Path to the first map txt file. }"
        "{ @target_map | | Path to the second map txt file. }"
        "{ config c | | Path to the YAML config file. }"
        "{ help h usage ? | | Print this help message. }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Map Alignment v1.0.0");

    if (parser.has("help")) {
      parser.printMessage();
      return 0;
    }

    // Load config file
    auto config_path = parser.get<std::string>("config");
    if (config_path.empty()) {
      std::cerr << "ERROR: Config file path not provided." << std::endl;
      return 1;
    }

    std::cout << "Loading configuration from: " << config_path << std::endl;
    config.loadFromFile(config_path);

    // Read map file names
    auto source_map_path = parser.get<std::string>("@source_map");
    if (source_map_path.empty())
      source_map_path = config.getOrDefault<std::string>("source", "");

    auto target_map_path = parser.get<std::string>("@target_map");
    if (target_map_path.empty())
      target_map_path = config.getOrDefault<std::string>("target", "");

    if (source_map_path.empty() || target_map_path.empty()) {
      std::cerr << "ERROR: No path provided for the map files." << std::endl;
      return 1;
    }

    // Load maps
    Map::Ptr source_map(new Map);
    source_map->load(source_map_path);

    Map::Ptr target_map(new Map);
    target_map->load(target_map_path);

    std::cout << "Loaded " << source_map->cloud->size() << " source landmarks and "
              << target_map->cloud->size() << " target landmarks." << std::endl;

    // Apply initial displacement to source map
    {
      auto displacement_config = config["displacement"];
      auto x = displacement_config.getOrDefault<double>("x", 0.0);
      auto y = displacement_config.getOrDefault<double>("y", 0.0);
      auto z = displacement_config.getOrDefault<double>("z", 0.0);
      auto roll = displacement_config.getOrDefault<double>("roll", 0.0);
      auto pitch = displacement_config.getOrDefault<double>("pitch", 0.0);
      auto yaw = displacement_config.getOrDefault<double>("yaw", 0.0);

      mrpt::poses::CPose3D transformation(
          x, y, z, mrpt::DEG2RAD(yaw), mrpt::DEG2RAD(pitch), mrpt::DEG2RAD(roll)
      );

      auto matrix = transformation.getHomogeneousMatrixVal<mrpt::math::CMatrixDouble44>();
      Eigen::Matrix4d transformation_matrix = matrix.asEigen();

      pcl::transformPointCloud(*source_map->cloud, *source_map->cloud, transformation_matrix);
    }

    auto experiments = run_experiments(source_map, target_map);
    alignment_visualization(experiments);
    score_histories_visualization(experiments);

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << mrpt::exception_to_str(e) << std::endl;
    return -1;
  }
}
