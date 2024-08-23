//! \example tutorial-mb-generic-tracker-ukf.cpp
#include <cstdlib>
#include <visp3/core/vpConfig.h>
#include <visp3/core/vpIoTools.h>
#include <visp3/core/vpUnscentedKalmanPose.h>
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

int main(int argc, char **argv)
{
#if defined(VISP_HAVE_OPENCV) && defined(VISP_HAVE_DISPLAY)
  try {
    std::string opt_videoname = "teabox-long.mp4";
    int opt_tracker = vpMbGenericTracker::EDGE_TRACKER;
    double opt_stdevP0 = 0.0001;
    double opt_stdevQ = 0.001;
    double opt_stdevR = 0.000001;

    for (int i = 0; i < argc; i++) {
      if (std::string(argv[i]) == "--name" && i + 1 < argc) {
        opt_videoname = std::string(argv[i + 1]);
      }
      else if (std::string(argv[i]) == "--tracker" && i + 1 < argc) {
        opt_tracker = atoi(argv[i + 1]);
      }
      else if (std::string(argv[i]) == "--P0" && i + 1 < argc) {
        opt_stdevP0 = std::atof(argv[i + 1]);
      }
      else if (std::string(argv[i]) == "--Q" && i + 1 < argc) {
        opt_stdevQ = std::atof(argv[i + 1]);
      }
      else if (std::string(argv[i]) == "--R" && i + 1 < argc) {
        opt_stdevR = std::atof(argv[i + 1]);
      }
      else if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
        std::cout << "\nUsage: " << argv[0] << " [--name <video name>] [--tracker <1=egde|2=keypoint|3=hybrid>]"
          << " [--P0 <stdev_P0>] [--Q <stdev_Q>] [--R <stdev_R>]"
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
    vpMatrix P0 = Id * opt_stdevP0 * opt_stdevP0;
    vpMatrix Q = Id * opt_stdevQ * opt_stdevQ;
    vpMatrix R = Id * opt_stdevR * opt_stdevR;
    double alphaPred = 0.1;
    const double dt = 0.040; // 40ms <=> 25Hz
    vpUnscentedKalmanPose::State X0;
    Id.eye(3);
    vpMatrix R_ukfm = Id * opt_stdevR * opt_stdevR;
    vpUnscentedKalmanPose ukfm(Q, R_ukfm, std::vector<double>(3, alphaPred), X0, P0,
                                          vpUnscentedKalmanPose::fSE3, vpUnscentedKalmanPose::hSE3, vpUnscentedKalmanPose::phiSE3,
                                          vpUnscentedKalmanPose::phiinvSE3);

    vpMouseButton::vpMouseButtonType button;
    bool stepbystep = true;
    int frame_cpt = 0;
    vpHomogeneousMatrix cMo_prev, cMo_filt;
    bool doesContinue = true;
    vpCameraParameters cam;
    vpPlot plot(2, 700, 700, I.getCols() + 50, 50, "Comparing 3D position");
    plot.initGraph(0, 3);
    plot.initGraph(1, 3);
    plot.setTitle(0, "cMo");
    plot.setLegend(0, 0, "X");
    plot.setLegend(0, 1, "Y");
    plot.setLegend(0, 2, "Z");
    plot.setTitle(1, "UKFM");
    plot.setLegend(1, 0, "X");
    plot.setLegend(1, 1, "Y");
    plot.setLegend(1, 2, "Z");
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
      }
      else {
        v = vpExponentialMap::inverse(cMo_prev * cMo.inverse(), dt);
      }
      ukfm.filter(v, vpUnscentedKalmanPose::asPositionVector(cMo), dt);
      cMo_filt = ukfm.getState();

      plot.plot(0, t, cMo.getTranslationVector());
      plot.plot(1, t, cMo_filt.getTranslationVector());

      //! [Display]
      tracker.getCameraParameters(cam);
      tracker.display(I, cMo, cam, vpColor::red, 2);
      //! [Display]
      vpDisplay::displayFrame(I, cMo, cam, 0.025, vpColor::none, 3);
      vpDisplay::displayFrame(I, cMo_filt, cam, 0.025, vpColor::yellow, 3);
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
      t += dt;
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
