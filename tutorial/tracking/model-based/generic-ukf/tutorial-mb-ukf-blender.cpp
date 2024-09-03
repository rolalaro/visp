//! \example tutorial-mb-generic-tracker-rgbd-blender.cpp
#include <iostream>

#include <visp3/core/vpConfig.h>
#include <visp3/core/vpDisplay.h>
#include <visp3/core/vpIoTools.h>
#include <visp3/core/vpParticleFilter.h>
#include <visp3/core/vpUnscentedKalmanPose.h>
#include <visp3/core/vpXmlParserCamera.h>
#include <visp3/gui/vpDisplayGDI.h>
#include <visp3/gui/vpDisplayOpenCV.h>
#include <visp3/gui/vpDisplayX.h>
#include <visp3/io/vpImageIo.h>
#include <visp3/mbt/vpMbGenericTracker.h>

#if defined(VISP_HAVE_OPENCV) && defined(HAVE_OPENCV_IMGCODECS) && defined(VISP_HAVE_PUGIXML)
#ifdef ENABLE_VISP_NAMESPACE
using namespace VISP_NAMESPACE_NAME;
#endif

namespace
{
bool read_data(unsigned int cpt, const std::string &video_color_images, const std::string &video_depth_images,
               bool opt_disable_depth, const std::string &video_ground_truth,
               vpImage<unsigned char> &I, vpImage<uint16_t> &I_depth_raw,
               unsigned int &depth_width, unsigned int &depth_height,
               std::vector<vpColVector> &pointcloud, const vpCameraParameters &cam_depth,
               vpHomogeneousMatrix &cMo_ground_truth)
{
  char buffer[FILENAME_MAX];
  // Read color
  snprintf(buffer, FILENAME_MAX, video_color_images.c_str(), cpt);
  std::string filename_color = buffer;

  if (!vpIoTools::checkFilename(filename_color)) {
    std::cerr << "Cannot read: " << filename_color << std::endl;
    return false;
  }
  vpImageIo::read(I, filename_color);

  if (!opt_disable_depth) {
    // Read depth
    snprintf(buffer, FILENAME_MAX, video_depth_images.c_str(), cpt);
    std::string filename_depth = buffer;

    if (!vpIoTools::checkFilename(filename_depth)) {
      std::cerr << "Cannot read: " << filename_depth << std::endl;
      return false;
    }
    cv::Mat depth_raw = cv::imread(filename_depth, cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
    if (depth_raw.empty()) {
      std::cerr << "Cannot read: " << filename_depth << std::endl;
      return false;
    }

    depth_width = static_cast<unsigned int>(depth_raw.cols);
    depth_height = static_cast<unsigned int>(depth_raw.rows);
    I_depth_raw.resize(depth_height, depth_width);
    pointcloud.resize(depth_width * depth_height);

    for (int i = 0; i < depth_raw.rows; i++) {
      for (int j = 0; j < depth_raw.cols; j++) {
        I_depth_raw[i][j] = static_cast<uint16_t>(32767.5f * depth_raw.at<cv::Vec3f>(i, j)[0]);
        double x = 0.0, y = 0.0;
        // Manually limit the field of view of the depth camera
        double Z = depth_raw.at<cv::Vec3f>(i, j)[0] > 2.0f ? 0.0 : static_cast<double>(depth_raw.at<cv::Vec3f>(i, j)[0]);
        vpPixelMeterConversion::convertPoint(cam_depth, j, i, x, y);
        size_t idx = static_cast<size_t>(i * depth_raw.cols + j);
        pointcloud[idx].resize(3);
        pointcloud[idx][0] = x * Z;
        pointcloud[idx][1] = y * Z;
        pointcloud[idx][2] = Z;
      }
    }
  }

  // Read ground truth
  snprintf(buffer, FILENAME_MAX, video_ground_truth.c_str(), cpt);
  std::string filename_pose = buffer;

  cMo_ground_truth.load(filename_pose);

  return true;
}

/*!
 * \brief Converts a PF state into a vpHomogeneousMatrix.
 * \param[in] x PF state.
 * \return The corresponding pose.
 */
vpHomogeneousMatrix stateToHomogeneousMatrix(const vpColVector &x)
{
  vpTranslationVector cTo(x[0], x[1], x[2]);
  vpThetaUVector c_Tu_o(x[3], x[4], x[5]);
  return vpHomogeneousMatrix(cTo, c_Tu_o);
}

/*!
 * \brief Converts a vpHomogeneousMatrix into a PF state .
 * \param[in] cMo The pose.
 * \return The corresponding PF state.
 */
vpColVector homogeneousMatrixToState(const vpHomogeneousMatrix &cMo)
{
  vpColVector x(6);
  vpTranslationVector cTo = cMo.getTranslationVector();
  vpThetaUVector c_Tu_o = cMo.getThetaUVector();
  x[0] = cTo[0];
  x[1] = cTo[1];
  x[2] = cTo[2];
  x[3] = c_Tu_o[0];
  x[4] = c_Tu_o[1];
  x[5] = c_Tu_o[2];
  return x;
}

/**
 * @brief Compute the weighted mean of poses.
 * Translations and rotations are taken care of separately.
 * Weighted mean orientation is computed following [this answer](https://stackoverflow.com/a/27410865).
 *
 * \param[in] w_i eights
 * \param[in] cMos Homogeneous matrices.
 * \return vpHomogeneousMatrix Weighted mean pose.
 */
vpHomogeneousMatrix computeMean(const std::vector<double> &w_i, const std::vector<vpHomogeneousMatrix> &cMos)
{
  const unsigned int nbMat = w_i.size();
  vpMatrix Q(4, nbMat);
  vpColVector meanT(3, 0.);
  for (unsigned int i = 0; i < nbMat; ++i) {
    double w = w_i[i];
    vpTranslationVector cTo = cMos[i].getTranslationVector();
    meanT += w * vpColVector({ cTo[0], cTo[1], cTo[2] });
    vpQuaternionVector qi = vpQuaternionVector(cMos[i].getRotationMatrix());
    double sqrtW = std::sqrt(w);
    Q[0][i] = sqrtW * qi[0];
    Q[1][i] = sqrtW * qi[1];
    Q[2][i] = sqrtW * qi[2];
    Q[3][i] = sqrtW * qi[3];
  }

  vpMatrix QQt = Q * Q.transpose();
  vpColVector eval;
  vpMatrix evec;
  QQt.eigenValues(eval, evec);
  vpColVector meanQvec = evec.getCol(evec.getCols() - 1);
  meanQvec /= meanQvec.frobeniusNorm();
  vpQuaternionVector meanQ(meanQvec);
  vpHomogeneousMatrix mean_cMo;
  mean_cMo.build(vpTranslationVector(meanT), meanQ);
  return mean_cMo;
}

vpColVector weightedMean(const std::vector<vpColVector> &states, const std::vector<double> &wis, const vpParticleFilter<vpHomogeneousMatrix>::vpStateAddFunction &/**/)
{
  const unsigned int nbMat = wis.size();
  std::vector<vpHomogeneousMatrix> poses;
  for (unsigned int i = 0; i < nbMat; ++i) {
    poses.push_back(stateToHomogeneousMatrix(states[i]));
  }
  vpHomogeneousMatrix meanPose = computeMean(wis, poses);
  return homogeneousMatrixToState(meanPose);
}

// void testMean()
// {
//   vpGaussRand rngT(0.1, 0);
//   vpGaussRand rngR(vpMath::rad(10), 0);
//   const double N = 10000;
//   vpTranslationVector cTo(0.1, 0.2, 0.3);
//   const double rx = vpMath::rad(10);
//   const double ry = vpMath::rad(10);
//   const double rz = vpMath::rad(10);
//   vpRxyzVector Rxyz(rx, ry, rz);
//   vpRotationMatrix cRo(Rxyz);
//   vpHomogeneousMatrix meanGT(cTo, cRo);
//   std::vector<double> wis;
//   std::vector<vpHomogeneousMatrix> cMos;
//   const double common_weight = 1. / static_cast<double>(N);
//   for (unsigned int i = 0; i < N; ++i) {
//     vpTranslationVector cTo_noisy = cTo + vpTranslationVector(rngT(), rngT(), rngT());
//     vpRxyzVector Rxyz_noisy(rx + rngR(), ry + rngR(), rz + rngR());
//     vpHomogeneousMatrix cMo_noisy(cTo_noisy, vpRotationMatrix(Rxyz_noisy));
//     cMos.push_back(cMo_noisy);
//     wis.push_back(common_weight);
//   }
//   vpHomogeneousMatrix meanCpted = computeMean(wis, cMos);
//   std::cout << "GT = " << std::endl << meanGT << std::endl;
//   std::cout << "Computed = " << std::endl << meanCpted << std::endl;
//   vpHomogeneousMatrix cgtTccpted = meanGT * meanCpted.inverse();
//   std::cout << "cgtTccpted = " << cgtTccpted.getTranslationVector().t() << "\t|cTc| = " << cgtTccpted.getTranslationVector().frobeniusNorm() << std::endl;
//   std::cout << "cgtRccpted = " << vpMath::deg(vpRxyzVector(cgtTccpted.getRotationMatrix())).t() << std::endl;
// }

/**
 * \brief Class that permits to project a particle forward in time.
 */
class vpProcessFunctor
{
public:
  vpProcessFunctor()
    : m_vel(6, 0.)
  {

  }

  vpColVector fx(const vpColVector &state, const double &dt)
  {
    vpHomogeneousMatrix cprevMo = stateToHomogeneousMatrix(state);
    const unsigned int nbSteps = 10;
    double dtSub = dt / static_cast<double>(nbSteps);
    vpHomogeneousMatrix updatedPose;
    for (unsigned int i = 0; i < nbSteps; ++i) {
      vpHomogeneousMatrix cprevMc = vpExponentialMap::direct(m_vel, dtSub);
      updatedPose = cprevMc.inverse() * cprevMo;
      cprevMo = updatedPose;
    }
    return homogeneousMatrixToState(updatedPose);
  }

  void setVel(const vpColVector &vel)
  {
    m_vel = vel;
  }
private:
  vpColVector m_vel;
};

/**
 * \brief Class that permits to compute the likelihood of a particle.
 */
class vpLikelihoodFunctor
{
public:
  /**
   * \brief Construct a new vpLikelihoodFunctor object.
   *
   * \param[in] translation_stdev The standard deviation of the translational error for the likelihood computation. A particle that is
   * 3. * likelihood_stdev further than the measurements will have a weight of 0.
   * \param[in] orientation_stdev The standard deviation of the rotational error for the likelihood computation. A particle that is
   * 3. * likelihood_stdev further than the measurements will have a weight of 0.
   */
  vpLikelihoodFunctor(const double &translation_stdev, const double &orientation_stdev)
  {
    double sigmaDistanceSquared = translation_stdev * translation_stdev;
    m_constantDenominatorTranslation = 1. / std::sqrt(2. * M_PI * sigmaDistanceSquared);
    m_constantExpDenominatorTranslation = -1. / (2. * sigmaDistanceSquared);
    double sigmaOrientationSquared = orientation_stdev * orientation_stdev;
    m_constantDenominatorOrientation = 1. / std::sqrt(2. * M_PI * sigmaOrientationSquared);
    m_constantExpDenominatorOrientation = -1. / (2. * sigmaOrientationSquared);
  }

  //! [Likelihood_function]
  /**
   * \brief Compute the likelihood of a particle compared to the measurements.
   * The likelihood equals zero if the particle is completely different of
   * the measurements and equals one if it matches completely.
   * The chosen likelihood is a mixture of Gaussian functions that penalizes the distance
   * and the orientation error  between the pose that represents a particle and
   * the pose that is measured.
   *
   * \param[in] x The particle.
   * \param[in] meas The measurement vector. meas[2i] = u_i meas[2i + 1] = v_i .
   * \return double The likelihood of the particle.
   */
  double likelihood(const vpColVector &x, const vpHomogeneousMatrix &cMo_meas)
  {
    double likelihood = 0.;
    vpHomogeneousMatrix cMo_particle = stateToHomogeneousMatrix(x);
    vpHomogeneousMatrix cmeasMcparticle = cMo_meas * cMo_particle.inverse();
    double distance = cmeasMcparticle.getTranslationVector().frobeniusNorm();
    double angleError = std::sqrt(cmeasMcparticle.getThetaUVector().sumSquare());
    // Compute the likelihood from the mean error
    double likelihoodTranslation = std::exp(m_constantExpDenominatorTranslation * distance)  * m_constantDenominatorTranslation;
    double likelihoodOrientation = std::exp(m_constantExpDenominatorOrientation * angleError)  * m_constantDenominatorOrientation;
    likelihood = 0.5 * likelihoodTranslation + 0.5 * likelihoodOrientation;
    likelihood = std::min(likelihood, 1.0); // Clamp to have likelihood <= 1.
    likelihood = std::max(likelihood, 0.); // Clamp to have likelihood >= 0.
    return likelihood;
  }
  //! [Likelihood_function]
private:
  double m_constantDenominatorTranslation; // Denominator of the Gaussian function used for the translation error for the likelihood computation.
  double m_constantExpDenominatorTranslation; // Denominator of the exponential of the Gaussian function used for the translation error for the likelihood computation.
  double m_constantDenominatorOrientation; // Denominator of the Gaussian function used for the orientation error for the likelihood computation.
  double m_constantExpDenominatorOrientation; // Denominator of the exponential of the Gaussian function used for the orientation error for the likelihood computation.
};
} // namespace

//! [CLI]
struct SoftwareArguments
{
  // --- Main loop parameters---
  static const int SOFTWARE_CONTINUE = 42;
  bool m_useDisplay; //!< If true, activate the plot and the renderer if VISP_HAVE_DISPLAY is defined.
  unsigned int m_nbStepsWarmUp; //!< Number of steps for the warmup phase.
  unsigned int m_nbSteps; //!< Number of steps for the main loop.
  std::string m_data_path;
  std::string m_model_path;
  unsigned int m_first_frame;
  double m_dt;
  // ---MBT params---
  int m_meMode;
  int m_kltMode;
  int m_normalsMode;
  int m_denseMode;
  bool m_disable_depth;
  bool m_disable_klt;
  bool m_display_ground_truth;
  bool m_step_by_step;
// --- UKFM parameters---
  double m_stdevP0;
  double m_stdevQ;
  double m_stdevR;
  double m_alphaPred;
// --- PF parameters---
  unsigned int m_N; //!< The number of particles.
  double m_maxDistanceForLikelihood; //!< The maximum allowed distance between a particle and the measurement, leading to a likelihood equal to 0..
  double m_maxOrientationErrorForLikelihood; //!< The maximum orientation error allowed between a particle and the measurement, leading to a likelihood equal to 0..
  double m_ampliMaxtX; //!< Amplitude max of the noise for the state component corresponding to the X coordinate.
  double m_ampliMaxtY; //!< Amplitude max of the noise for the state component corresponding to the Y coordinate.
  double m_ampliMaxtZ; //!< Amplitude max of the noise for the state component corresponding to the Z coordinate.
  double m_ampliMaxtuX; //!< Amplitude max of the noise for the state component corresponding to the Theta u X coordinate.
  double m_ampliMaxtuY; //!< Amplitude max of the noise for the state component corresponding to the Theta u Y coordinate.
  double m_ampliMaxtuZ; //!< Amplitude max of the noise for the state component corresponding to the Theta u Z coordinate.
  long m_seedPF; //!< Seed for the random generators of the PF.
  int m_nbThreads; //!< Number of thread to use in the Particle Filter.

  SoftwareArguments()
    : m_useDisplay(true)
    , m_nbStepsWarmUp(200)
    , m_nbSteps(300)
    , m_data_path("data/teabox")
    , m_model_path("model/teabox")
    , m_first_frame(1)
    , m_dt(0.04)
    , m_meMode(1)
    , m_kltMode(1)
    , m_normalsMode(0)
    , m_denseMode(1)
    , m_disable_depth(false)
    , m_disable_klt(false)
    , m_display_ground_truth(false)
    , m_step_by_step(false)
    , m_stdevP0(0.001)
    , m_stdevQ(0.000001)
    , m_stdevR(0.000001)
    , m_alphaPred(0.1)
    , m_N(500)
    , m_maxDistanceForLikelihood(0.05)
    , m_maxOrientationErrorForLikelihood(vpMath::rad(5))
    , m_ampliMaxtX(0.005)
    , m_ampliMaxtY(0.005)
    , m_ampliMaxtZ(0.005)
    , m_ampliMaxtuX(vpMath::rad(2.5))
    , m_ampliMaxtuY(vpMath::rad(2.5))
    , m_ampliMaxtuZ(vpMath::rad(2.5))
    , m_seedPF(4224)
    , m_nbThreads(10)
  { }

  int parseArgs(const int argc, const char *argv[])
  {
    int i = 1;

    while (i < argc) {
      std::string arg(argv[i]);
      if ((arg == "--nb-steps-main") && ((i+1) < argc)) {
        m_nbSteps = std::atoi(argv[i + 1]);
        ++i;
      }
      else if ((arg == "--nb-steps-warmup") && ((i+1) < argc)) {
        m_nbStepsWarmUp = std::atoi(argv[i + 1]);
        ++i;
      }
      else if (std::string(argv[i]) == "--data-path" && i + 1 < argc) {
        m_data_path = std::string(argv[i + 1]);
        ++i;
      }
      else if (std::string(argv[i]) == "--model-path" && i + 1 < argc) {
        m_model_path = std::string(argv[i + 1]);
        ++i;
      }
      else if (std::string(argv[i]) == "--depth-dense-mode" && i + 1 < argc) {
        m_denseMode = static_cast<unsigned int>(atoi(argv[i + 1]));
        ++i;
      }
      else if (std::string(argv[i]) == "--depth-normals-mode" && i + 1 < argc) {
        m_normalsMode = static_cast<unsigned int>(atoi(argv[i + 1]));
        ++i;
      }
      else if (std::string(argv[i]) == "--me-mode" && i + 1 < argc) {
        m_meMode = static_cast<unsigned int>(atoi(argv[i + 1]));
        ++i;
      }
      else if (std::string(argv[i]) == "--klt-mode" && i + 1 < argc) {
        m_kltMode = static_cast<unsigned int>(atoi(argv[i + 1]));
        ++i;
      }
      else if (std::string(argv[i]) == "--display-ground-truth") {
        m_display_ground_truth = true;
      }
      else if (std::string(argv[i]) == "--step-by-step") {
        m_step_by_step = true;
      }
      else if (std::string(argv[i]) == "--first-frame" && i + 1 < argc) {
        m_first_frame = static_cast<unsigned int>(atoi(argv[i + 1]));
        ++i;
      }
      else if (std::string(argv[i]) == "--P0" && i + 1 < argc) {
        m_stdevP0 = std::atof(argv[i + 1]);
        ++i;
      }
      else if (std::string(argv[i]) == "--Q" && i + 1 < argc) {
        m_stdevQ = std::atof(argv[i + 1]);
        ++i;
      }
      else if (std::string(argv[i]) == "--R" && i + 1 < argc) {
        m_stdevR = std::atof(argv[i + 1]);
        ++i;
      }
      else if (std::string(argv[i]) == "--dt" && i + 1 < argc) {
        m_dt = std::atof(argv[i + 1]);
        ++i;
      }
      else if (std::string(argv[i]) == "--alpha" && i + 1 < argc) {
        m_alphaPred = std::atof(argv[i + 1]);
        ++i;
      }
      else if ((arg == "--max-distance-likelihood") && ((i+1) < argc)) {
        m_maxDistanceForLikelihood = std::atof(argv[i + 1]);
        ++i;
      }
      else if ((arg == "--max-orientation-likelihood") && ((i+1) < argc)) {
        m_maxOrientationErrorForLikelihood = vpMath::rad(std::atof(argv[i + 1]));
        ++i;
      }
      else if (((arg == "-N") || (arg == "--nb-particles")) && ((i+1) < argc)) {
        m_N = std::atoi(argv[i + 1]);
        ++i;
      }
      else if ((arg == "--seed") && ((i+1) < argc)) {
        m_seedPF = std::atoi(argv[i + 1]);
        ++i;
      }
      else if ((arg == "--nb-threads") && ((i+1) < argc)) {
        m_nbThreads = std::atoi(argv[i + 1]);
        ++i;
      }
      else if ((arg == "--ampli-max-tX") && ((i+1) < argc)) {
        m_ampliMaxtX = std::atof(argv[i + 1]);
        ++i;
      }
      else if ((arg == "--ampli-max-tY") && ((i+1) < argc)) {
        m_ampliMaxtY = std::atof(argv[i + 1]);
        ++i;
      }
      else if ((arg == "--ampli-max-tZ") && ((i+1) < argc)) {
        m_ampliMaxtZ = std::atof(argv[i + 1]);
        ++i;
      }
      else if ((arg == "--ampli-max-rX") && ((i+1) < argc)) {
        m_ampliMaxtuX = vpMath::rad(std::atof(argv[i + 1]));
        ++i;
      }
      else if ((arg == "--ampli-max-rY") && ((i+1) < argc)) {
        m_ampliMaxtuY = vpMath::rad(std::atof(argv[i + 1]));
        ++i;
      }
      else if ((arg == "--ampli-max-rZ") && ((i+1) < argc)) {
        m_ampliMaxtuZ = vpMath::rad(std::atof(argv[i + 1]));
        ++i;
      }
      else if (arg == "-d") {
        m_useDisplay = false;
      }
      else if ((arg == "-h") || (arg == "--help")) {
        printUsage(std::string(argv[0]));
        SoftwareArguments defaultArgs;
        defaultArgs.printDetails();
        return 0;
      }
      else {
        std::cout << "WARNING: unrecognised argument \"" << arg << "\"";
        if (i + 1 < argc) {
          std::cout << " with associated value(s) { ";
          int nbValues = 0;
          int j = i + 1;
          bool hasToRun = true;
          while ((j < argc) && hasToRun) {
            std::string nextValue(argv[j]);
            if (nextValue.find("--") == std::string::npos) {
              std::cout << nextValue << " ";
              ++nbValues;
            }
            else {
              hasToRun = false;
            }
            ++j;
          }
          std::cout << "}" << std::endl;
          i += nbValues;
        }
      }
      ++i;
    }
    m_disable_depth = m_denseMode == 0 && m_normalsMode == 0;
    return SOFTWARE_CONTINUE;
  }

private:
  void printUsage(const std::string &softName)
  {
    std::cout << "SYNOPSIS" << std::endl;
    std::cout << "  " << softName << " [--nb-steps-main <uint>] [--nb-steps-warmup <uint>]" << std::endl;
    std::cout << " [--data-path <path>] [--model-path <path>] [--first-frame <index>] [--depth-dense-mode <0|1>] " << std::endl;
    std::cout << " [--depth-normals-mode <0|1>] [--me-mode <0|1>] [--klt-mode <0|1>] [--step-by-step] [--display-ground-truth]" << std::endl;
    std::cout << "  [--tracker <1=egde|2=keypoint|3=hybrid>] [--dt <double>]" << std::endl;
    std::cout << "  [--P0 <double>] [--Q <double>] [--R <double>] [--alpha <double>]" << std::endl;
    std::cout << "  [--max-distance-likelihood <double>] [--max-orientation-likelihood <double>] [-N, --nb-particles <uint>] [--seed <int>] [--nb-threads <int>]" << std::endl;
    std::cout << "  [--ampli-max-tX <double>] [--ampli-max-tY <double>] [--ampli-max-tZ <double>]" << std::endl;
    std::cout << "  [--ampli-max-rX <double>] [--ampli-max-rY <double>] [--ampli-max-rZ <double>]" << std::endl;
    std::cout << "  [-d, --no-display] [-h]" << std::endl;
  }

  void printDetails()
  {
    std::cout << std::endl << std::endl;
    std::cout << "DETAILS" << std::endl;
    std::cout << " [Simu params]" << std::endl;
    std::cout << "  --nb-steps-main" << std::endl;
    std::cout << "    Number of steps in the main loop." << std::endl;
    std::cout << "    Default: " << m_nbSteps << std::endl;
    std::cout << std::endl;
    std::cout << "  --nb-steps-warmup" << std::endl;
    std::cout << "    Number of steps in the warmup loop." << std::endl;
    std::cout << "    Default: " << m_nbStepsWarmUp << std::endl;
    std::cout << std::endl;
    std::cout << "  --data-path <path>   Path to the data generated by Blender get_camera_pose_teabox.py" << std::endl;
    std::cout << "    Python script.";
    std::cout << "    Default: " << m_data_path << std::endl;
    std::cout << std::endl;
    std::cout << "  --model-path <path>   Path to the cad model and tracker settings." << std::endl;
    std::cout << "    Default: " << m_model_path << std::endl;
    std::cout << std::endl;
    std::cout << "  --first-frame <index>   First frame number to process." << std::endl;
    std::cout << "    Default: " << m_first_frame << std::endl;
    std::cout << std::endl;
    std::cout << "  --dt <seconds>" << std::endl;
    std::cout << "    The timestep between two frames, in seconds." << std::endl;
    std::cout << "    Default: " << m_dt << std::endl;
    std::cout << std::endl;
    std::cout << "  --step-by-step  Flag to enable step by step mode." << std::endl;
    std::cout << std::endl;
    std::cout << "  --display-ground-truth  Flag to enable displaying ground truth." << std::endl;
    std::cout << "    When this flag is enabled, there is no tracking. This flag is useful" << std::endl;
    std::cout << "    to validate the ground truth over the rendered images." << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << " [MBT params]" << std::endl;
    std::cout << "  --depth-dense-mode  Whether to use dense depth features (0 = off, 1 = on). default: 1" << std::endl;
    std::cout << std::endl;
    std::cout << "  --depth-normals-mode  Whether to use normal depth features (0 = off, 1 = on). default: 0" << std::endl;
    std::cout << std::endl;
    std::cout << "  --me-mode  Whether to use moving edge features (0 = off, 1 = on). default: 1" << std::endl;
    std::cout << std::endl;
    std::cout << "  --klt-mode  Whether to use KLT features (0 = off, 1 = on). Requires OpenCV. default: 1" << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << " [UKF params]" << std::endl;
    std::cout << "  --P0" << std::endl;
    std::cout << "    The standard deviation of the initial process covariance matrix." << std::endl;
    std::cout << "    Default: " << m_stdevP0 << std::endl;
    std::cout << std::endl;
    std::cout << "  --Q" << std::endl;
    std::cout << "    The standard deviation of the process covariance matrix." << std::endl;
    std::cout << "    Default: " << m_stdevQ << std::endl;
    std::cout << std::endl;
    std::cout << "  --R" << std::endl;
    std::cout << "    The standard deviation of the measurement covariance matrix." << std::endl;
    std::cout << "    Default: " << m_stdevR << std::endl;
    std::cout << std::endl;
    std::cout << "  --alpha" << std::endl;
    std::cout << "    The spreading factor of the chi points." << std::endl;
    std::cout << "    Default: " << m_alphaPred << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << " [PF params]" << std::endl;
    std::cout << "  --max-distance-likelihood" << std::endl;
    std::cout << "    Maximum distance between a particle with the measurements." << std::endl;
    std::cout << "    Above this value, the likelihood of the particle is 0." << std::endl;
    std::cout << "    Default: " << m_maxDistanceForLikelihood << std::endl;
    std::cout << std::endl;
    std::cout << "  --max-orientation-likelihood" << std::endl;
    std::cout << "    Maximum orientation error, in degrees, between a particle with the measurements." << std::endl;
    std::cout << "    Above this value, the likelihood of the particle is 0." << std::endl;
    std::cout << "    Default: " << vpMath::deg(m_maxOrientationErrorForLikelihood) << std::endl;
    std::cout << std::endl;
    std::cout << "  -N, --nb-particles" << std::endl;
    std::cout << "    Number of particles of the Particle Filter." << std::endl;
    std::cout << "    Default: " << m_N << std::endl;
    std::cout << std::endl;
    std::cout << "  --seed" << std::endl;
    std::cout << "    Seed to initialize the Particle Filter." << std::endl;
    std::cout << "    Use a negative value makes to use the current timestamp instead." << std::endl;
    std::cout << "    Default: " << m_seedPF << std::endl;
    std::cout << std::endl;
    std::cout << "  --nb-threads" << std::endl;
    std::cout << "    Set the number of threads to use in the Particle Filter (only if OpenMP is available)." << std::endl;
    std::cout << "    Use a negative value to use the maximum number of threads instead." << std::endl;
    std::cout << "    Default: " << m_nbThreads << std::endl;
    std::cout << std::endl;
    std::cout << "  --ampli-max-tX" << std::endl;
    std::cout << "    Maximum amplitude of the noise added t the translational part of a particle along the X-axis." << std::endl;
    std::cout << "    Default: " << m_ampliMaxtX << std::endl;
    std::cout << std::endl;
    std::cout << "  --ampli-max-tY" << std::endl;
    std::cout << "    Maximum amplitude of the noise added to the translational part of a particle along the Y-axis." << std::endl;
    std::cout << "    Default: " << m_ampliMaxtY << std::endl;
    std::cout << std::endl;
    std::cout << "  --ampli-max-tZ" << std::endl;
    std::cout << "    Maximum amplitude of the noise added to the translational part of a particle along the Z-axis." << std::endl;
    std::cout << "    Default: " << m_ampliMaxtZ << std::endl;
    std::cout << std::endl;
    std::cout << "  --ampli-max-rX" << std::endl;
    std::cout << "    Maximum amplitude of the noise added to the rotational part of a particle around the X-axis." << std::endl;
    std::cout << "    Default: " << vpMath::deg(m_ampliMaxtuX) << std::endl;
    std::cout << std::endl;
    std::cout << "  --ampli-max-rY" << std::endl;
    std::cout << "    Maximum amplitude of the noise added to the rotational part of a particle around the Y-axis." << std::endl;
    std::cout << "    Default: " << vpMath::deg(m_ampliMaxtuY) << std::endl;
    std::cout << std::endl;
    std::cout << "  --ampli-max-rZ" << std::endl;
    std::cout << "    Maximum amplitude of the noise added to the rotational part of a particle around the Z-axis." << std::endl;
    std::cout << "    Default: " << vpMath::deg(m_ampliMaxtuZ) << std::endl;
    std::cout << std::endl;
    std::cout << " [Other params]" << std::endl;
    std::cout << "  -d, --no-display" << std::endl;
    std::cout << "    Deactivate display." << std::endl;
    std::cout << "    Default: display is ";
#ifdef VISP_HAVE_DISPLAY
    std::cout << "ON" << std::endl;
#else
    std::cout << "OFF" << std::endl;
#endif
    std::cout << std::endl;
    std::cout << "  -h, --help" << std::endl;
    std::cout << "    Display this help." << std::endl;
    std::cout << std::endl;
  }
};
//! [CLI]

int main(int argc, const char **argv)
{
  SoftwareArguments args;
  int returnCode = args.parseArgs(argc, argv);
  if (returnCode != SoftwareArguments::SOFTWARE_CONTINUE) {
    return returnCode;
  }

  std::string video_color_images = vpIoTools::createFilePath(args.m_data_path, "color/%04d_L.jpg");
  std::string video_depth_images = vpIoTools::createFilePath(args.m_data_path, "depth/Image%04d_R.exr");
  std::string ground_truth = vpIoTools::createFilePath(args.m_data_path, "ground-truth/Camera_L_%04d.txt");
  std::string extrinsic_file = vpIoTools::createFilePath(args.m_data_path, "depth_M_color.txt");
  std::string color_camera_name = "Camera_L";
  std::string depth_camera_name = "Camera_R";
  std::string color_intrinsic_file = vpIoTools::createFilePath(args.m_data_path, color_camera_name + ".xml");
  std::string depth_intrinsic_file = vpIoTools::createFilePath(args.m_data_path, depth_camera_name + ".xml");
  std::string mbt_config_color = vpIoTools::createFilePath(args.m_model_path, "teabox_color.xml");
  std::string mbt_config_depth = vpIoTools::createFilePath(args.m_model_path, "teabox_depth.xml");
  std::string mbt_cad_model = vpIoTools::createFilePath(args.m_model_path, "teabox.cao");
  std::string mbt_init_file = vpIoTools::createFilePath(args.m_model_path, "teabox.init");

  std::cout << "Input data" << std::endl;
  std::cout << "  Color images    : " << video_color_images << std::endl;
  std::cout << "  Depth images    : " << (args.m_disable_depth ? "Disabled" : video_depth_images) << std::endl;
  std::cout << "  Extrinsics      : " << (args.m_disable_depth ? "Disabled" : extrinsic_file) << std::endl;
  std::cout << "  Color intrinsics: " << color_intrinsic_file << std::endl;
  std::cout << "  Depth intrinsics: " << (args.m_disable_depth ? "Disabled" : depth_intrinsic_file) << std::endl;
  std::cout << "  Ground truth    : " << ground_truth << std::endl;
  std::cout << "Tracker settings" << std::endl;
  std::cout << "  Color config    : " << mbt_config_color << std::endl;
  std::cout << "  Depth config    : " << mbt_config_depth << std::endl;
  std::cout << "  CAD model       : " << mbt_cad_model << std::endl;
  std::cout << "  First frame     : " << args.m_first_frame << std::endl;
  std::cout << "  Step by step    : " << args.m_step_by_step << std::endl;
  if (args.m_display_ground_truth) {
    std::cout << "  Ground truth is used to project the cad model (no tracking)" << std::endl;
  }
  else {
    std::cout << "  Init file       : " << mbt_init_file << std::endl;
    std::cout << "  Features        : moving-edges " << (args.m_disable_klt ? "" : "+ keypoints") << (args.m_disable_depth ? "" : " + depth") << std::endl;
  }


  std::vector<int> tracker_types;
  int colorTracker = 0;


  if (args.m_meMode == 1) {
    colorTracker |= vpMbGenericTracker::EDGE_TRACKER;
  }
  if (args.m_kltMode == 1) {
#if defined(VISP_HAVE_OPENCV) && defined(HAVE_OPENCV_IMGPROC) && defined(HAVE_OPENCV_VIDEO)
    colorTracker |= vpMbGenericTracker::KLT_TRACKER;
#else
    std::cerr << "Warning: keypoints cannot be used as features since ViSP is not built with OpenCV 3rd party" << std::endl;
#endif
  }

  if (colorTracker == 0) {
    std::cerr << "You should use at least one type of color feature. If OpenCV is not installed, KLT features are disabled" << std::endl;
    return EXIT_FAILURE;
  }

  tracker_types.push_back(colorTracker);

  if (!args.m_disable_depth) {
    int depthTracker = 0;
    if (args.m_denseMode == 1) {
      depthTracker |= vpMbGenericTracker::DEPTH_DENSE_TRACKER;
    }
    if (args.m_normalsMode == 1) {
      depthTracker |= vpMbGenericTracker::DEPTH_NORMAL_TRACKER;
    }

    tracker_types.push_back(depthTracker);
  }

  vpMbGenericTracker tracker(tracker_types);
  if (!args.m_disable_depth) {
    tracker.loadConfigFile(mbt_config_color, mbt_config_depth, true);
  }
  else {
    tracker.loadConfigFile(mbt_config_color);

  }
  tracker.loadModel(mbt_cad_model);
  vpCameraParameters cam_color, cam_depth;

  // Update intrinsics camera parameters from Blender generated data
  vpXmlParserCamera p;
  if (p.parse(cam_color, color_intrinsic_file, color_camera_name, vpCameraParameters::perspectiveProjWithoutDistortion)
     != vpXmlParserCamera::SEQUENCE_OK) {
    std::cout << "Cannot found intrinsics for camera " << color_camera_name << std::endl;
  }
  if (p.parse(cam_depth, depth_intrinsic_file, depth_camera_name, vpCameraParameters::perspectiveProjWithoutDistortion)
     != vpXmlParserCamera::SEQUENCE_OK) {
    std::cout << "Cannot found intrinsics for camera " << depth_camera_name << std::endl;
  }

  if (!args.m_disable_depth)
    tracker.setCameraParameters(cam_color, cam_depth);
  else
    tracker.setCameraParameters(cam_color);

  // Reload intrinsics from tracker (useless)
  if (!args.m_disable_depth)
    tracker.getCameraParameters(cam_color, cam_depth);
  else
    tracker.getCameraParameters(cam_color);
  tracker.setDisplayFeatures(true);
  std::cout << "cam_color:\n" << cam_color << std::endl;

  if (!args.m_disable_depth)
    std::cout << "cam_depth:\n" << cam_depth << std::endl;

  vpImage<uint16_t> I_depth_raw;
  vpImage<unsigned char> I, I_depth;
  unsigned int depth_width = 0, depth_height = 0;
  std::vector<vpColVector> pointcloud;
  vpHomogeneousMatrix cMo_ground_truth;

  unsigned int frame_cpt = args.m_first_frame;
  read_data(frame_cpt, video_color_images, video_depth_images, args.m_disable_depth, ground_truth,
            I, I_depth_raw, depth_width, depth_height, pointcloud, cam_depth, cMo_ground_truth);
  vpImageConvert::createDepthHistogram(I_depth_raw, I_depth);

#if defined(VISP_HAVE_X11)
  vpDisplayX d1, d2;
#elif defined(VISP_HAVE_GDI)
  vpDisplayGDI d1, d2;
#elif defined (HAVE_OPENCV_HIGHGUI)
  vpDisplayOpenCV d1, d2;
#endif

  d1.init(I, 0, 0, "Color image");
  if (!args.m_disable_depth) {
    d2.init(I_depth, static_cast<int>(I.getWidth()), 0, "Depth image");
  }

  vpHomogeneousMatrix depth_M_color;
  if (!args.m_disable_depth) {
    depth_M_color.load(extrinsic_file);
    tracker.setCameraTransformationMatrix("Camera2", depth_M_color);
    std::cout << "depth_M_color:\n" << depth_M_color << std::endl;
  }

  if (args.m_display_ground_truth) {
    tracker.initFromPose(I, cMo_ground_truth); // I and I_depth must be the same size when using depth features!
  }
  else {
    tracker.initClick(I, mbt_init_file, true); // I and I_depth must be the same size when using depth features!
  }

  vpMatrix Id;
  Id.eye(6);
  vpMatrix P0 = Id * args.m_stdevP0 * args.m_stdevP0;
  vpMatrix Q = Id * args.m_stdevQ * args.m_stdevQ;
  vpMatrix R = Id * args.m_stdevR * args.m_stdevR;
  double alphaPred = 0.1;
  const double dt = 0.040; // 40ms <=> 25Hz
  vpUnscentedKalmanPose::State X0;
  Id.eye(3);
  vpMatrix R_ukfm = Id * args.m_stdevR * args.m_stdevR;
  vpUnscentedKalmanPose ukfm(Q, R_ukfm, std::vector<double>(3, alphaPred), X0, P0,
                                        vpUnscentedKalmanPose::fSE3, vpUnscentedKalmanPose::hSE3, vpUnscentedKalmanPose::phiSE3,
                                        vpUnscentedKalmanPose::phiinvSE3);


  //! [Particle_filter]
  vpLikelihoodFunctor likelihoodFunctor(args.m_maxDistanceForLikelihood, args.m_maxOrientationErrorForLikelihood);
  vpProcessFunctor processFunctor;
  using std::placeholders::_1;
  using std::placeholders::_2;
  vpParticleFilter<vpHomogeneousMatrix>::vpProcessFunction processFunc = std::bind(&vpProcessFunctor::fx, &processFunctor, _1, _2);
  vpParticleFilter<vpHomogeneousMatrix>::vpLikelihoodFunction likelihoodFunc = std::bind(&vpLikelihoodFunctor::likelihood, &likelihoodFunctor, _1, _2);
  vpParticleFilter<vpHomogeneousMatrix>::vpResamplingConditionFunction checkResamplingFunc = vpParticleFilter<vpHomogeneousMatrix>::simpleResamplingCheck;
  vpParticleFilter<vpHomogeneousMatrix>::vpResamplingFunction resamplingFunc = vpParticleFilter<vpHomogeneousMatrix>::simpleImportanceResampling;
  vpParticleFilter<vpHomogeneousMatrix>::vpFilterFunction filteringFunc = weightedMean;
  std::vector<double> stdevsPF { args.m_ampliMaxtX / 3., args.m_ampliMaxtY / 3., args.m_ampliMaxtZ / 3., args.m_ampliMaxtuX / 3., args.m_ampliMaxtuY / 3., args.m_ampliMaxtuZ / 3. };
  vpParticleFilter<vpHomogeneousMatrix> pfFilter(args.m_N, stdevsPF, args.m_seedPF, args.m_nbThreads);
  //! [Particle_filter]

  vpHomogeneousMatrix cMo_prev, cMo_filt;
  bool isNotInitialized = true;
  try {
    bool quit = false;
    while (!quit && read_data(frame_cpt, video_color_images, video_depth_images, args.m_disable_depth,
                              ground_truth, I, I_depth_raw, depth_width, depth_height, pointcloud, cam_depth,
                              cMo_ground_truth)) {
      vpImageConvert::createDepthHistogram(I_depth_raw, I_depth);
      vpDisplay::display(I);
      vpDisplay::display(I_depth);

      if (args.m_display_ground_truth) {
        tracker.initFromPose(I, cMo_ground_truth); // I and I_depth must be the same size when using depth features!
      }
      else {
        if (!args.m_disable_depth) {
          std::map<std::string, const vpImage<unsigned char> *> mapOfImages;
          std::map<std::string, const std::vector<vpColVector> *> mapOfPointClouds;
          std::map<std::string, unsigned int> mapOfPointCloudWidths;
          std::map<std::string, unsigned int> mapOfPointCloudHeights;

          mapOfImages["Camera1"] = &I;
          mapOfPointClouds["Camera2"] = &pointcloud;
          mapOfPointCloudWidths["Camera2"] = depth_width;
          mapOfPointCloudHeights["Camera2"] = depth_height;
          tracker.track(mapOfImages, mapOfPointClouds, mapOfPointCloudWidths, mapOfPointCloudHeights);
        }
        else {
          tracker.track(I);
        }
      }

      vpHomogeneousMatrix cMo = tracker.getPose();
      vpColVector v(6, 0.);
      if (isNotInitialized) {
        ukfm.setX0(cMo);
        vpColVector cXo = homogeneousMatrixToState(cMo);
        pfFilter.init(cXo, processFunc, likelihoodFunc, checkResamplingFunc, resamplingFunc, filteringFunc);
        isNotInitialized = false;
      }
      else {
        v = vpExponentialMap::inverse(cMo_prev * cMo.inverse(), dt);
        std::cout << "v = " << v.transpose() << std::endl;
      }
      ukfm.filter(v, vpUnscentedKalmanPose::asPositionVector(cMo), dt);
      cMo_filt = ukfm.getState();

      processFunctor.setVel(v);
      pfFilter.filter(cMo, args.m_dt);
      vpColVector Xfilt_pf = pfFilter.computeFilteredState();
      vpHomogeneousMatrix cMo_pf = stateToHomogeneousMatrix(Xfilt_pf);

      std::cout << "\nFrame: " << frame_cpt << std::endl;
      if (!args.m_display_ground_truth)
        std::cout << "cMo:\n" << cMo << std::endl;
      std::cout << "cMo_filt:\n" << cMo_filt << std::endl;
      std::cout << "cMo_PF:\n" << cMo_pf << std::endl;
      std::cout << "cMo ground truth:\n" << cMo_ground_truth << std::endl;
      if (!args.m_disable_depth) {
        tracker.display(I, I_depth, cMo, depth_M_color * cMo, cam_color, cam_depth, vpColor::red, 2);
        vpDisplay::displayFrame(I_depth, depth_M_color * cMo, cam_depth, 0.05, vpColor::none, 2);
      }
      else {
        tracker.display(I, cMo, cam_color, vpColor::red, 2);
      }

      vpDisplay::displayFrame(I, cMo, cam_color, 0.05, vpColor::none, 2);
      vpDisplay::displayFrame(I, cMo_filt, cam_color, 0.05, vpColor::yellow, 3);
      vpDisplay::displayFrame(I, cMo_pf, cam_color, 0.05, vpColor::purple, 3);
      std::ostringstream oss;
      oss << "Frame: " << frame_cpt;
      vpDisplay::setTitle(I, oss.str());
      if (args.m_step_by_step) {
        vpDisplay::displayText(I, 20, 10, "Left click to trigger next step", vpColor::red);
        vpDisplay::displayText(I, 40, 10, "Right click to quit step-by-step mode", vpColor::red);
      }
      else {
        vpDisplay::displayText(I, 20, 10, "Left click to trigger step-by-step mode", vpColor::red);
        vpDisplay::displayText(I, 40, 10, "Right click to exit...", vpColor::red);
      }
      if (!args.m_display_ground_truth) {
        {
          std::stringstream ss;
          ss << "Nb features: " << tracker.getError().size();
          vpDisplay::displayText(I, I.getHeight() - 50, 20, ss.str(), vpColor::red);
        }
        {
          std::stringstream ss;
          ss << "Features: edges " << tracker.getNbFeaturesEdge() << ", klt " << tracker.getNbFeaturesKlt()
            << ", depth " << tracker.getNbFeaturesDepthDense();
          vpDisplay::displayText(I, I.getHeight() - 30, 20, ss.str(), vpColor::red);
        }
      }

      vpDisplay::flush(I);
      vpDisplay::flush(I_depth);

      // Button 1: start step by step if not enabled from command line option
      // Button 2: enables step by step mode
      // Button 3: ends step by step mode if enabled
      //           quit otherwise
      vpMouseButton::vpMouseButtonType button;
      if (vpDisplay::getClick(I, button, args.m_step_by_step)) {
        if (button == vpMouseButton::button1 && args.m_step_by_step == false) {
          args.m_step_by_step = true;
        }
        else if (button == vpMouseButton::button3 && args.m_step_by_step == true) {
          args.m_step_by_step = false;
        }
        else if (button == vpMouseButton::button3 && args.m_step_by_step == false) {
          quit = true;
        }
        else if (button == vpMouseButton::button2) {
          args.m_step_by_step = true;
        }
      }

      frame_cpt++;
      cMo_prev = cMo;
    }

    vpDisplay::flush(I);
    vpDisplay::getClick(I);
  }
  catch (std::exception &e) {
    std::cerr << "Catch exception: " << e.what() << std::endl;
  }

  return EXIT_SUCCESS;
}
#else
int main()
{
  std::cout << "To run this tutorial, ViSP should be built with OpenCV and pugixml libraries." << std::endl;
  return EXIT_SUCCESS;
}
#endif
