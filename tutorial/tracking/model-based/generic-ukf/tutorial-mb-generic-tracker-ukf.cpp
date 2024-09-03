//! \example tutorial-mb-generic-tracker-ukf.cpp
#include <cstdlib>
#include <visp3/core/vpConfig.h>
#include <visp3/core/vpCameraParameters.h>
#include <visp3/core/vpIoTools.h>
#include <visp3/core/vpUnscentedKalmanPose.h>
#include <visp3/core/vpParticleFilter.h>
#include <visp3/gui/vpDisplayGDI.h>
#include <visp3/gui/vpDisplayOpenCV.h>
#include <visp3/gui/vpDisplayX.h>
#include <visp3/gui/vpPlot.h>
#include <visp3/io/vpImageIo.h>
//! [Include]
#include <visp3/mbt/vpMbGenericTracker.h>
//! [Include]
#include <visp3/io/vpVideoReader.h>

#ifdef ENABLE_VISP_NAMESPACE
using namespace VISP_NAMESPACE_NAME;
#endif

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

//! [CLI]
struct SoftwareArguments
{
  // --- Main loop parameters---
  static const int SOFTWARE_CONTINUE = 42;
  bool m_useDisplay; //!< If true, activate the plot and the renderer if VISP_HAVE_DISPLAY is defined.
  unsigned int m_nbStepsWarmUp; //!< Number of steps for the warmup phase.
  unsigned int m_nbSteps; //!< Number of steps for the main loop.
  std::string m_videoname; //!< Name of the video sequence.
  int m_tracker = vpMbGenericTracker::EDGE_TRACKER; //!< Type of MBT tracker
  double m_dt;
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
    , m_videoname("teabox-long.mp4")
    , m_tracker(vpMbGenericTracker::EDGE_TRACKER)
    , m_dt(0.04)
    , m_stdevP0(0.0001)
    , m_stdevQ(0.001)
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
      else if (std::string(argv[i]) == "--name" && i + 1 < argc) {
        m_videoname = std::string(argv[i + 1]);
        ++i;
      }
      else if (std::string(argv[i]) == "--tracker" && i + 1 < argc) {
        m_tracker = atoi(argv[i + 1]);
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
    return SOFTWARE_CONTINUE;
  }

private:
  void printUsage(const std::string &softName)
  {
    std::cout << "SYNOPSIS" << std::endl;
    std::cout << "  " << softName << " [--nb-steps-main <uint>] [--nb-steps-warmup <uint>]" << std::endl;
    std::cout << "  [--name <string>] [--tracker <1=egde|2=keypoint|3=hybrid>] [--dt <double>]" << std::endl;
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
    std::cout << "  --name <video name>" << std::endl;
    std::cout << "    The path to the video sequence." << std::endl;
    std::cout << "    Default: " << m_videoname << std::endl;
    std::cout << std::endl;
    std::cout << "  --tracker <1=egde|2=keypoint|3=hybrid>" << std::endl;
    std::cout << "    The type of MBT tracker to use." << std::endl;
    std::cout << "    Default: " << m_tracker << std::endl;
    std::cout << std::endl;
    std::cout << "  --dt <seconds>" << std::endl;
    std::cout << "    The timestep between two frames, in seconds." << std::endl;
    std::cout << "    Default: " << m_dt << std::endl;
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

int main(const int argc, const char **argv)
{
  // testMean();
  // return 0;
#if defined(VISP_HAVE_OPENCV) && defined(VISP_HAVE_DISPLAY)
  try {
    SoftwareArguments args;
    int returnCode = args.parseArgs(argc, argv);
    if (returnCode != SoftwareArguments::SOFTWARE_CONTINUE) {
      return returnCode;
    }

    if (args.m_tracker < 1 || args.m_tracker > 3) {
      std::cerr << "Wrong tracker type. Correct values are: "
        "1=egde|2=keypoint|3=hybrid."
        << std::endl;
      return EXIT_SUCCESS;
    }

    std::string parentname = vpIoTools::getParent(args.m_videoname);
    std::string objectname = vpIoTools::getNameWE(args.m_videoname);

    if (!parentname.empty()) {
      objectname = parentname + "/" + objectname;
    }

    std::cout << "Video name: " << args.m_videoname << std::endl;
    std::cout << "Tracker requested config files: " << objectname << ".[init, cao]" << std::endl;
    std::cout << "Tracker optional config files: " << objectname << ".[ppm]" << std::endl;

    //! [Image]
    vpImage<unsigned char> I;
    //! [Image]

    vpVideoReader g;
    g.setFileName(args.m_videoname);
    g.open(I);

#if defined(VISP_HAVE_X11)
    vpDisplayX display;
#elif defined(VISP_HAVE_GDI)
    vpDisplayGDI display;
#elif defined(HAVE_OPENCV_HIGHGUI)
    vpDisplayOpenCV display;
#endif
    display.init(I, 100, 100, "Model-based tracker");

    //! [Constructor]
    vpMbGenericTracker tracker(1, args.m_tracker);
    //! [Constructor]

#if !defined(VISP_HAVE_MODULE_KLT)
    if (args.m_tracker >= 2) {
      std::cout << "KLT and hybrid model-based tracker are not available since visp_klt module is missing"
        << std::endl;
      return EXIT_SUCCESS;
    }
#endif

    //! [Set parameters]

#if defined(VISP_HAVE_PUGIXML)
    //! [Load config file]
    tracker.loadConfigFile(objectname + ".xml");
    //! [Load config file]
#else
    // Corresponding parameters manually set to have an example code
    if (args.m_tracker == 1 || args.m_tracker == 3) {
      vpMe me;
      me.setMaskSize(5);
      me.setMaskNumber(180);
      me.setRange(8);
      me.setLikelihoodThresholdType(vpMe::NORMALIZED_THRESHOLD);
      me.setThreshold(20);
      me.setMu1(0.5);
      me.setMu2(0.5);
      me.setSampleStep(4);
      tracker.setMovingEdge(me);
    }

#if defined(VISP_HAVE_MODULE_KLT) && defined(VISP_HAVE_OPENCV) && defined(HAVE_OPENCV_IMGPROC) && defined(HAVE_OPENCV_VIDEO)
    if (args.m_tracker == 2 || args.m_tracker == 3) {
      vpKltOpencv klt_settings;
      tracker.setKltMaskBorder(5);
      klt_settings.setMaxFeatures(300);
      klt_settings.setWindowSize(5);
      klt_settings.setQuality(0.015);
      klt_settings.setMinDistance(8);
      klt_settings.setHarrisFreeParameter(0.01);
      klt_settings.setBlockSize(3);
      klt_settings.setPyramidLevels(3);
      tracker.setKltOpencv(klt_settings);
    }
#endif

    {
      //! [Set camera parameters]
      vpCameraParameters cam;
      cam.initPersProjWithoutDistortion(839.21470, 839.44555, 325.66776, 243.69727);
      tracker.setCameraParameters(cam);
      //! [Set camera parameters]
    }
#endif
    //! [Set parameters]

    //! [Load cao]
    tracker.loadModel(objectname + ".cao");
    //! [Load cao]
    //! [Set display features]
    tracker.setDisplayFeatures(true);
    //! [Set display features]
    //! [Init]
    tracker.initClick(I, objectname + ".init", true);
    //! [Init]

    vpMatrix Id;
    Id.eye(6);
    vpMatrix P0 = Id * args.m_stdevP0 * args.m_stdevP0;
    vpMatrix Q = Id * args.m_stdevQ * args.m_stdevQ;
    vpMatrix R = Id * args.m_stdevR * args.m_stdevR;
    vpUnscentedKalmanPose::State X0;
    Id.eye(3);
    vpMatrix R_ukfm = Id * args.m_stdevR * args.m_stdevR;
    vpUnscentedKalmanPose ukfm(Q, R_ukfm, std::vector<double>(3, args.m_alphaPred), X0, P0,
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

    vpMouseButton::vpMouseButtonType button;
    bool stepbystep = true;
    int frame_cpt = 0;
    vpHomogeneousMatrix cMo_prev, cMo_filt;
    bool doesContinue = true;
    vpCameraParameters cam;
    vpPlot plot(3, 700, 700, I.getCols() + 50, 50, "Comparing 3D position");
    plot.initGraph(0, 3);
    plot.initGraph(1, 3);
    plot.initGraph(2, 3);
    plot.setTitle(0, "cMo");
    plot.setLegend(0, 0, "X");
    plot.setLegend(0, 1, "Y");
    plot.setLegend(0, 2, "Z");
    plot.setTitle(1, "UKFM");
    plot.setLegend(1, 0, "X");
    plot.setLegend(1, 1, "Y");
    plot.setLegend(1, 2, "Z");
    plot.setTitle(2, "PF");
    plot.setLegend(2, 0, "X");
    plot.setLegend(2, 1, "Y");
    plot.setLegend(2, 2, "Z");
    double t = 0.;
    while ((!g.end()) &&  doesContinue) {
      g.acquire(I);

      vpDisplay::display(I);
      //! [Track]
      tracker.track(I);
      //! [Track]
      //! [Get pose]
      vpHomogeneousMatrix cMo;
      tracker.getPose(cMo);
      //! [Get pose]

      vpColVector v(6, 0.);
      if (frame_cpt == 0) {
        ukfm.setX0(cMo);
        vpColVector cXo = homogeneousMatrixToState(cMo);
        pfFilter.init(cXo, processFunc, likelihoodFunc, checkResamplingFunc, resamplingFunc, filteringFunc);
      }
      else {
        v = vpExponentialMap::inverse(cMo_prev * cMo.inverse(), args.m_dt);
      }
      ukfm.filter(v, vpUnscentedKalmanPose::asPositionVector(cMo), args.m_dt);
      cMo_filt = ukfm.getState();

      processFunctor.setVel(v);
      pfFilter.filter(cMo, args.m_dt);
      vpColVector Xfilt_pf = pfFilter.computeFilteredState();
      vpHomogeneousMatrix cMo_pf = stateToHomogeneousMatrix(Xfilt_pf);

      plot.plot(0, t, cMo.getTranslationVector());
      plot.plot(1, t, cMo_filt.getTranslationVector());
      plot.plot(2, t, cMo_pf.getTranslationVector());

      //! [Display]
      tracker.getCameraParameters(cam);
      tracker.display(I, cMo, cam, vpColor::red, 2);
      //! [Display]
      vpDisplay::displayFrame(I, cMo, cam, 0.025, vpColor::none, 3);
      vpDisplay::displayFrame(I, cMo_filt, cam, 0.025, vpColor::yellow, 3);
      vpDisplay::displayFrame(I, cMo_pf, cam, 0.025, vpColor::purple, 3);
      vpDisplay::displayText(I, 10, 10, "A right click to exit...", vpColor::red);
      vpDisplay::displayText(I, 30, 10, "A middle click to switch to " + (stepbystep ? std::string("auto") : std::string("step-by-step")) + " mode", vpColor::red);
      if (stepbystep) {
        vpDisplay::displayText(I, 50, 10, "A left click to display the next frame.", vpColor::red);
      }
      vpDisplay::flush(I);

      if (vpDisplay::getClick(I, button, stepbystep)) {
        switch (button) {
        case vpMouseButton::button2:
          stepbystep = stepbystep xor true;
          break;
        case vpMouseButton::button3:
          doesContinue = false;
          break;
        default:
          break;
        }
      }

      ++frame_cpt;
      t += args.m_dt;
      cMo_prev = cMo;
    }
    vpDisplay::display(I);
    vpDisplay::displayText(I, 10, 10, "Any click to exit...", vpColor::red);
    vpDisplay::displayFrame(I, cMo_filt, cam, 0.025, vpColor::yellow, 3);
    vpDisplay::flush(I);
    vpDisplay::getClick(I);
  }
  catch (const vpException &e) {
    std::cerr << "Catch a ViSP exception: " << e.what() << std::endl;
  }

  return EXIT_SUCCESS;
#else
  (void)argc;
  (void)argv;
  std::cout << "Install OpenCV and rebuild ViSP to use this example." << std::endl;
  return EXIT_SUCCESS;
#endif
}
