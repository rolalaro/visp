//! \example tutorial-mb-generic-tracker-ukf.cpp
#include <cstdlib>
#include <visp3/core/vpConfig.h>
#include <visp3/core/vpIoTools.h>
#include <visp3/core/vpUnscentedKalmanPose.h>
#include <visp3/gui/vpDisplayGDI.h>
#include <visp3/gui/vpDisplayOpenCV.h>
#include <visp3/gui/vpDisplayX.h>
#include <visp3/io/vpImageIo.h>
//! [Include]
#include <visp3/mbt/vpMbGenericTracker.h>
//! [Include]
#include <visp3/io/vpVideoReader.h>

#include "UKFM.h"

#ifdef ENABLE_VISP_NAMESPACE
using namespace VISP_NAMESPACE_NAME;
#endif

vpColVector asColVector(const vpTranslationVector &t)
{
  vpColVector tAsVec(3);
  tAsVec[0] = t[0];
  tAsVec[1] = t[1];
  tAsVec[2] = t[2];
  return tAsVec;
}

UKFM::State phi(const UKFM::State &xi, const vpColVector &epsilon, const double &dt)
{
  UKFM::State expEpsilon = vpExponentialMap::direct(epsilon, dt);
  return xi * expEpsilon;
}

vpColVector phiinv(const UKFM::State &state, const UKFM::State &hat_state, const double &dt)
{
  return vpExponentialMap::inverse(state.inverse() * hat_state, dt);
}

UKFM::State f(const UKFM::State &state, const vpColVector &omega, const vpColVector &w, const double &dt)
{
  return state * vpExponentialMap::direct(omega + w, dt);
}

vpColVector h(const UKFM::State &state)
{
  vpTranslationVector p = state.getTranslationVector();
  vpColVector pAsVec = asColVector(p);
  return pAsVec;
}

int main(int argc, char **argv)
{
#if defined(VISP_HAVE_OPENCV)
  try {
    std::string opt_videoname = "teabox-long.mp4";
    int opt_tracker = vpMbGenericTracker::EDGE_TRACKER;

    for (int i = 0; i < argc; i++) {
      if (std::string(argv[i]) == "--name" && i + 1 < argc)
        opt_videoname = std::string(argv[i + 1]);
      else if (std::string(argv[i]) == "--tracker" && i + 1 < argc)
        opt_tracker = atoi(argv[i + 1]);
      else if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
        std::cout << "\nUsage: " << argv[0] << " [--name <video name>] [--tracker <1=egde|2=keypoint|3=hybrid>]"
          << " [--help] [-h]\n"
          << std::endl;
        return EXIT_SUCCESS;
      }
    }

    if (opt_tracker < 1 || opt_tracker > 3) {
      std::cerr << "Wrong tracker type. Correct values are: "
        "1=egde|2=keypoint|3=hybrid."
        << std::endl;
      return EXIT_SUCCESS;
    }

    std::string parentname = vpIoTools::getParent(opt_videoname);
    std::string objectname = vpIoTools::getNameWE(opt_videoname);

    if (!parentname.empty()) {
      objectname = parentname + "/" + objectname;
    }

    std::cout << "Video name: " << opt_videoname << std::endl;
    std::cout << "Tracker requested config files: " << objectname << ".[init, cao]" << std::endl;
    std::cout << "Tracker optional config files: " << objectname << ".[ppm]" << std::endl;

    //! [Image]
    vpImage<unsigned char> I;
    //! [Image]

    vpVideoReader g;
    g.setFileName(opt_videoname);
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
    vpMbGenericTracker tracker(1, opt_tracker);
    //! [Constructor]

#if !defined(VISP_HAVE_MODULE_KLT)
    if (opt_tracker >= 2) {
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
    if (opt_tracker == 1 || opt_tracker == 3) {
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
    if (opt_tracker == 2 || opt_tracker == 3) {
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

    vpColVector omega0(6, 0.);
    vpMatrix Id;
    Id.eye(6);
    const double stdevX0 = 0.1;
    const double stdevQ = 0.001;
    const double stdevR = 0.1;
    vpMatrix P0 = Id * stdevX0 * stdevX0;
    vpMatrix Q = Id * stdevQ * stdevQ;
    vpMatrix R = Id * stdevR * stdevR;
    double alphaPred = 0.1;
    double alphaUpdate = alphaPred;
    vpColVector muNoiseMeas(6, 0.);
    vpUnscentedKalmanPose ukf(Q, R, muNoiseMeas, alphaPred, alphaUpdate);
    const double dt = 0.040; // 40ms <=> 25Hz
    UKFM::State X0;
    Id.eye(3);
    vpMatrix R_ukfm = Id * stdevR * stdevR;
    UKFM ukf_ukfmImplem(UKFM::ProcessFunction(f), UKFM::ObservationFunction(h), UKFM::RetractationFunction(phi),
                          UKFM::InverseRetractationFunction(phiinv), Q, R_ukfm, std::vector<double>(3, alphaPred), X0, P0);

    vpMouseButton::vpMouseButtonType button;
    bool stepbystep = true;
    int frame_cpt = 0;
    vpHomogeneousMatrix cMo_prev, cMo_filt;
    vpHomogeneousMatrix ukfm_cMo_prev, ukfm_cMo_filt;
    vpColVector ukfm_omega;
    bool doesContinue = true;
    vpCameraParameters cam;
    while ((!g.end()) && doesContinue) {
      g.acquire(I);
      vpDisplay::display(I);
      //! [Track]
      tracker.track(I);
      //! [Track]
      //! [Get pose]
      vpHomogeneousMatrix cMo;
      tracker.getPose(cMo);
      //! [Get pose]

      vpColVector z(6, 0.);
      if (frame_cpt == 0) {
        ukf.init(cMo, P0, omega0);
        ukf_ukfmImplem.setX0(cMo);
        ukfm_cMo_prev = cMo;
        ukfm_omega = omega0;
      }
      else {
        z = vpExponentialMap::inverse(cMo_prev * cMo.inverse(), dt);
      }
      ukf.filter(z, dt);
      cMo_filt = ukf.getXest();

      ukf_ukfmImplem.propagation(z, dt);
      ukf_ukfmImplem.update(asColVector(cMo.getTranslationVector()), dt);
      ukfm_cMo_filt = ukf_ukfmImplem.getXest();

      std::cout << "iter: " << frame_cpt << " cMo:\n" << cMo << std::endl;
      std::cout << "z = [" << z.transpose() << "]" << std::endl;
      std::cout << "\tcMo_ukfm:" << std::endl;
      for (unsigned int i = 0; i < 4; ++i) {
        std::cout << "\t\t";
        for (unsigned int j = 0; j < 3; ++j) {
          std::cout << ukfm_cMo_filt[i][j] << " ; ";
        }
        std::cout << ukfm_cMo_filt[i][3] << std::endl;
      }
      std::cout << "\t-> Error_ukfm:" << std::endl;
      vpHomogeneousMatrix cMcukfm = cMo * ukfm_cMo_filt.inverse();
      vpTranslationVector cTcukfm = cMcukfm.getTranslationVector();
      vpRxyzVector cRcukfm = vpRxyzVector(cMcukfm.getRotationMatrix());
      std::cout << "\t\t" << cTcukfm[0] * 1000. << " ; " << cTcukfm[1] * 1000. << " ; " << cTcukfm[2] * 1000. << " [mm]\n";
      std::cout << "\t\t" << vpMath::deg(cRcukfm).transpose() << " [deg]\n";
      std::cout << "\tcMo_filt:" << std::endl;
      for (unsigned int i = 0; i < 4; ++i) {
        std::cout << "\t\t";
        for (unsigned int j = 0; j < 3; ++j) {
          std::cout << cMo_filt[i][j] << " ; ";
        }
        std::cout << cMo_filt[i][3] << std::endl;
      }
      std::cout << "\t->Error:" << std::endl;
      vpHomogeneousMatrix cMcfilt = cMo * cMo_filt.inverse();
      vpTranslationVector cTcfilt = cMcfilt.getTranslationVector();
      vpRxyzVector cRcfilt = vpRxyzVector(cMcfilt.getRotationMatrix());
      std::cout << "\t\t" << cTcfilt[0] * 1000. << " ; " << cTcfilt[1] * 1000. << " ; " << cTcfilt[2] * 1000. << " [mm]\n";
      std::cout << "\t\t" << vpMath::deg(cRcfilt).transpose() << " [deg]\n";
      if (frame_cpt > 0) {
        vpColVector velocity = vpExponentialMap::inverse(cMo_prev * cMo.inverse(), dt);
        vpColVector filteredVelocity = ukf.getOmega();
        std::cout << "velocity = " <<  velocity.extract(0, 3).transpose() * 1000.f << " [mm/s]; " << vpMath::deg(velocity.extract(3, 3)).transpose() << " [deg/s]"  << std::endl;
        std::cout << "\tvelocity_filt = " <<  filteredVelocity.extract(0, 3).transpose() * 1000.f << " [mm/s]; " << vpMath::deg(filteredVelocity.extract(3, 3)).transpose() << " [deg/s]"  << std::endl;
        std::cout << "Error on the velocity:\n";
        vpColVector errorVel = velocity - filteredVelocity;
        std::cout << "\t\t" << errorVel[0] * 1000.f << " ; " << errorVel[1] * 1000.f << " ; " << errorVel[2] * 1000.f << " [mm/s]" << std::endl;
        std::cout << "\t\t" << vpMath::deg(errorVel[3]) << " ; " << vpMath::deg(errorVel[4]) << " ; " << vpMath::deg(errorVel[5]) << " [deg/s]" << std::endl;
      }

      //! [Display]
      tracker.getCameraParameters(cam);
      tracker.display(I, cMo, cam, vpColor::red, 2);
      //! [Display]
      vpDisplay::displayFrame(I, cMo, cam, 0.025, vpColor::none, 3);
      vpDisplay::displayFrame(I, cMo_filt, cam, 0.025, vpColor::yellow, 3);
      vpDisplay::displayFrame(I, ukfm_cMo_filt, cam, 0.025, vpColor::purple, 3);
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
      cMo_prev = cMo;
      ukfm_omega = vpExponentialMap::inverse(ukfm_cMo_prev * ukfm_cMo_filt.inverse(), dt);
      ukfm_cMo_prev = ukfm_cMo_filt;
    }
    vpDisplay::display(I);
    vpDisplay::displayText(I, 10, 10, "Any click to exit...", vpColor::red);
    vpDisplay::displayFrame(I, cMo_filt, cam, 0.025, vpColor::yellow, 3);
    vpDisplay::displayFrame(I, ukfm_cMo_filt, cam, 0.025, vpColor::purple, 3);
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
